// swift-tools-version:5.5
import PackageDescription

var exclude: [String] = []

#if os(Linux)
// Linux doesn't support CoreML, and will attempt to import the coreml source directory
exclude.append("coreml")
#endif

// Turn on the x86_64 vector ISA that ggml gates its fast paths behind.
//
// ggml.c only defines GGML_SIMD on x86 when __AVX__ is set, so without these
// flags every vector kernel on Intel falls back to scalar C, and __F16C__ being
// absent routes each F16 weight through a software lookup table. Xcode's x86_64
// baseline defines none of them. `-Xarch_x86_64` is what keeps this per slice:
// the flags reach only the x86_64 compile, so arm64 (NEON, already optimal) is
// byte-for-byte unchanged and the universal binary still builds.
//
// Safe floor: macOS 14 already requires an 8th-gen Coffee Lake / Amber Lake or
// Xeon-W Mac, and AVX2, FMA and F16C all shipped with Haswell in 2013. No Mac
// that can run this app lacks them.
let x86VectorISA = ["-Xarch_x86_64", "-mavx2", "-Xarch_x86_64", "-mfma", "-Xarch_x86_64", "-mf16c"]

let package = Package(
    name: "SwiftWhisper",
    products: [
        .library(name: "SwiftWhisper", targets: ["SwiftWhisper"])
    ],
    targets: [
        .target(name: "SwiftWhisper", dependencies: [.target(name: "whisper_cpp")]),
        .target(name: "whisper_cpp",
                exclude: exclude,
                cSettings: [
                    .define("GGML_USE_ACCELERATE", .when(platforms: [.macOS, .macCatalyst, .iOS])),
                    .define("WHISPER_USE_COREML", .when(platforms: [.macOS, .macCatalyst, .iOS])),
                    .define("WHISPER_COREML_ALLOW_FALLBACK", .when(platforms: [.macOS, .macCatalyst, .iOS])),
                    .unsafeFlags(x86VectorISA, .when(platforms: [.macOS, .macCatalyst]))
                ],
                cxxSettings: [
                    .unsafeFlags(x86VectorISA, .when(platforms: [.macOS, .macCatalyst]))
                ]),
        .testTarget(name: "WhisperTests", dependencies: [.target(name: "SwiftWhisper")], resources: [.copy("TestResources/")])
    ],
    cxxLanguageStandard: CXXLanguageStandard.cxx11
)

