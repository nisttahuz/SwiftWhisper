import Foundation
import whisper_cpp

/// A language whisper heard, by its code ("en", "es"), and how sure it is.
public struct DetectedLanguage: Equatable, Sendable {
    public let language: String
    public let probability: Float

    public init(language: String, probability: Float) {
        self.language = language
        self.probability = probability
    }
}

public class Whisper {
    private let whisperContext: OpaquePointer
    private var unmanagedSelf: Unmanaged<Whisper>?

    public var delegate: WhisperDelegate?
    public var params: WhisperParams
    public private(set) var inProgress = false

    /// What whisper heard in the last transcription when it detected the
    /// language itself, likeliest first. Empty when the language was given.
    public private(set) var detectedLanguages: [DetectedLanguage] = []

    /// Asked once per transcription, after whisper has detected the language and
    /// before it decodes. Return false and the transcription ends there with no
    /// segments, having cost the encoder pass alone. Not asked when the language
    /// was given. Called off the main thread.
    public var shouldDecode: (@Sendable ([DetectedLanguage]) -> Bool)?
    private var askedToDecode = false

    internal var frameCount: Int? // For progress calculation (value not in `whisper_state` yet)
    internal var cancelCallback: (() -> Void)?

    public init(fromFileURL fileURL: URL, withParams params: WhisperParams = .default) {
        self.whisperContext = fileURL.relativePath.withCString { whisper_init_from_file($0) }
        self.params = params
    }

    public init(fromData data: Data, withParams params: WhisperParams = .default) {
        var copy = data // Need to copy memory so we can gaurentee exclusive ownership over pointer

        self.whisperContext = copy.withUnsafeMutableBytes { whisper_init_from_buffer($0.baseAddress!, data.count) }
        self.params = params
    }

    deinit {
        whisper_free(whisperContext)
    }

    private func prepareCallbacks() {
        /*
         C-style callbacks can't capture any references in swift, so we'll convert `self`
         to a pointer which whisper passes back as the `user_data` argument.

         We can unwrap that and obtain a copy of self inside the callback.
         */
        cleanupCallbacks()
        let unmanagedSelf = Unmanaged.passRetained(self)
        self.unmanagedSelf = unmanagedSelf
        params.new_segment_callback_user_data = unmanagedSelf.toOpaque()
        params.encoder_begin_callback_user_data = unmanagedSelf.toOpaque()
        params.progress_callback_user_data = unmanagedSelf.toOpaque()

        // swiftlint:disable line_length
        params.new_segment_callback = { (ctx: OpaquePointer?, _: OpaquePointer?, newSegmentCount: Int32, userData: UnsafeMutableRawPointer?) in
        // swiftlint:enable line_length
            guard let ctx = ctx,
                  let userData = userData else { return }
            let whisper = Unmanaged<Whisper>.fromOpaque(userData).takeUnretainedValue()
            guard let delegate = whisper.delegate else { return }

            let segmentCount = whisper_full_n_segments(ctx)
            var newSegments: [Segment] = []
            newSegments.reserveCapacity(Int(newSegmentCount))

            let startIndex = segmentCount - newSegmentCount

            for index in startIndex..<segmentCount {
                guard let text = whisper_full_get_segment_text(ctx, index) else { continue }
                let startTime = whisper_full_get_segment_t0(ctx, index)
                let endTime = whisper_full_get_segment_t1(ctx, index)

                newSegments.append(.init(
                    startTime: Int(startTime) * 10, // Time is given in ms/10, so correct for that
                    endTime: Int(endTime) * 10,
                    text: String(Substring(cString: text))
                ))
            }

            DispatchQueue.main.async {
                delegate.whisper(whisper, didProcessNewSegments: newSegments, atIndex: Int(startIndex))
            }
        }

        params.encoder_begin_callback = { (_: OpaquePointer?, _: OpaquePointer?, userData: UnsafeMutableRawPointer?) in
            guard let userData = userData else { return true }
            let whisper = Unmanaged<Whisper>.fromOpaque(userData).takeUnretainedValue()

            if whisper.cancelCallback != nil {
                return false
            }

            if let shouldDecode = whisper.shouldDecode, !whisper.askedToDecode {
                whisper.askedToDecode = true
                let heard = whisper.readDetectedLanguages()
                if !heard.isEmpty, !shouldDecode(heard) { return false }
            }

            return true
        }

        // swiftlint:disable line_length
        params.progress_callback = { (_: OpaquePointer?, _: OpaquePointer?, progress: Int32, userData: UnsafeMutableRawPointer?) in
        // swiftlint:enable line_length
            guard let userData = userData else { return }
            let whisper = Unmanaged<Whisper>.fromOpaque(userData).takeUnretainedValue()

            DispatchQueue.main.async {
                whisper.delegate?.whisper(whisper, didUpdateProgress: Double(progress) / 100)
            }
        }
    }

    private func readDetectedLanguages() -> [DetectedLanguage] {
        (0...whisper_lang_max_id()).compactMap { id -> DetectedLanguage? in
            let probability = whisper_full_lang_prob(whisperContext, id)
            guard probability > 0, let code = whisper_lang_str(id) else { return nil }
            return DetectedLanguage(language: String(cString: code), probability: probability)
        }.sorted { $0.probability > $1.probability }
    }

    private func cleanupCallbacks() {
        guard let unmanagedSelf = unmanagedSelf else { return }

        unmanagedSelf.release()
        self.unmanagedSelf = nil
    }

    public func transcribe(audioFrames: [Float], completionHandler: @escaping (Result<[Segment], Error>) -> Void) {
        prepareCallbacks()

        let wrappedCompletionHandler: (Result<[Segment], Error>) -> Void = { result in
            self.cleanupCallbacks()
            completionHandler(result)
        }

        guard !inProgress else {
            wrappedCompletionHandler(.failure(WhisperError.instanceBusy))
            return
        }
        guard audioFrames.count > 0 else {
            wrappedCompletionHandler(.failure(WhisperError.invalidFrames))
            return
        }

        inProgress = true
        frameCount = audioFrames.count
        askedToDecode = false

        // The call keeps its own copy of the language. Setting `params.language`
        // frees the string it replaces, and whisper reads the pointer while it
        // runs, so a caller that set it meanwhile had whisper read freed memory.
        var callParams = params.whisperParams
        let language = strdup(callParams.language)
        callParams.language = UnsafePointer(language)

        DispatchQueue.global(qos: .userInitiated).async { [callParams] in
            whisper_full(self.whisperContext, callParams, audioFrames, Int32(audioFrames.count))
            free(language)
            self.detectedLanguages = self.readDetectedLanguages()

            let segmentCount = whisper_full_n_segments(self.whisperContext)

            var segments: [Segment] = []
            segments.reserveCapacity(Int(segmentCount))

            for index in 0..<segmentCount {
                guard let text = whisper_full_get_segment_text(self.whisperContext, index) else { continue }
                let startTime = whisper_full_get_segment_t0(self.whisperContext, index)
                let endTime = whisper_full_get_segment_t1(self.whisperContext, index)

                segments.append(
                    .init(
                        startTime: Int(startTime) * 10, // Correct for ms/10
                        endTime: Int(endTime) * 10,
                        text: String(Substring(cString: text))
                    )
                )
            }

            // Free for the next call before this one reports: a caller that
            // starts the next transcription from the completion handler could
            // arrive while `inProgress` was still set and be refused as busy.
            let cancelCallback = self.cancelCallback
            self.frameCount = nil
            self.cancelCallback = nil
            self.inProgress = false

            if let cancelCallback = cancelCallback {
                DispatchQueue.main.async {
                    // Should cancel callback be called after delegate and completionHandler?
                    cancelCallback()

                    let error = WhisperError.cancelled

                    self.delegate?.whisper(self, didErrorWith: error)
                    wrappedCompletionHandler(.failure(error))
                }
            } else {
                DispatchQueue.main.async {
                    self.delegate?.whisper(self, didCompleteWithSegments: segments)
                    wrappedCompletionHandler(.success(segments))
                }
            }
        }
    }

    public func cancel(completionHandler: @escaping () -> Void) throws {
        guard inProgress else { throw WhisperError.cancellationError(.notInProgress) }
        guard cancelCallback == nil else { throw WhisperError.cancellationError(.pendingCancellation)}

        cancelCallback = completionHandler
    }

    @available(iOS 13, macOS 10.15, watchOS 6.0, tvOS 13.0, *)
    public func transcribe(audioFrames: [Float]) async throws -> [Segment] {
        return try await withCheckedThrowingContinuation { cont in
            self.transcribe(audioFrames: audioFrames) { result in
                switch result {
                case .success(let segments):
                    cont.resume(returning: segments)
                case .failure(let error):
                    cont.resume(throwing: error)
                }
            }
        }
    }

    @available(iOS 13, macOS 10.15, watchOS 6.0, tvOS 13.0, *)
    public func cancel() async throws {
        return try await withCheckedThrowingContinuation { cont in
            do {
                try self.cancel {
                    cont.resume()
                }
            } catch {
                cont.resume(throwing: error)
            }
        }
    }
}
