import CiteGuardCore
import Foundation
import SwiftUI

@MainActor
final class CiteGuardViewModel: ObservableObject {
    @Published var documentText: String = ""
    @Published var mode: ResolutionMode = .cache
    @Published var results: [CitationResolution] = []
    @Published var privacyEntries: [PrivacyLogEntry] = []
    @Published var coverage: [CoverageSource] = [BundledAuthoritySnapshot().coverage]
    @Published var isScanning = false
    @Published var lastError: String?

    private let privacyLog = PrivacyLogStore()
    private let scanner: CitationScanner
    private let coverageClient: any CoverageFetching
    private let intake = DocumentIntake()

    init() {
        let resolver = CiteGuardAPIClient(
            apiKey: ProcessInfo.processInfo.environment["CITEGUARD_API_KEY"],
            privacyLog: privacyLog
        )
        scanner = CitationScanner(resolver: resolver, jurisdictionHint: .hk)
        coverageClient = CiteGuardCoverageClient()
    }

    func refreshCoverage() async {
        coverage = await coverageClient.fetchCoverage()
    }

    var summary: ScanSummary {
        ScanSummary(results: results)
    }

    func scanPastedText() async {
        await scan(text: documentText)
    }

    func importDocument(from url: URL) async {
        do {
            let hasAccess = url.startAccessingSecurityScopedResource()
            defer {
                if hasAccess {
                    url.stopAccessingSecurityScopedResource()
                }
            }
            documentText = try intake.extractText(from: url)
            await scan(text: documentText)
        } catch {
            lastError = error.localizedDescription
        }
    }

    func refreshPrivacyLog() async {
        privacyEntries = await privacyLog.allEntries()
    }

    private func scan(text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            results = []
            lastError = "Paste or open a document first."
            return
        }

        isScanning = true
        lastError = nil
        results = await scanner.scan(text: trimmed, mode: mode)
        await refreshPrivacyLog()
        isScanning = false
    }
}
