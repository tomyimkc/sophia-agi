// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "CiteGuard",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(name: "CiteGuardCore", targets: ["CiteGuardCore"])
    ],
    targets: [
        .target(
            name: "CiteGuardCore",
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "CiteGuardCoreTests",
            dependencies: ["CiteGuardCore"]
        )
    ]
)
