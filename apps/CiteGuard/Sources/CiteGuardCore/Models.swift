import Foundation

public enum CitationState: String, Codable, CaseIterable, Sendable {
    case confirmed
    case couldNotConfirm = "could_not_confirm"
    case unsupported

    public var displayTitle: String {
        switch self {
        case .confirmed:
            return "Confirmed in source"
        case .couldNotConfirm:
            return "Could not confirm"
        case .unsupported:
            return "Unsupported"
        }
    }

    public var zhTitle: String {
        switch self {
        case .confirmed:
            return "來源有紀錄"
        case .couldNotConfirm:
            return "未能確認"
        case .unsupported:
            return "暫不支援"
        }
    }
}

public enum JurisdictionHint: String, Codable, CaseIterable, Sendable {
    case hk = "HK"
    case uk = "UK"
    case us = "US"
    case auto
}

public enum ResolutionMode: String, Codable, CaseIterable, Sendable {
    case live
    case cache
}

public struct CitationCandidate: Identifiable, Hashable, Codable, Sendable {
    public let id: UUID
    public let raw: String
    public let normalized: String
    public let courtToken: String?
    public let jurisdiction: JurisdictionHint?
    public let isSupportedByLaunchScope: Bool

    public init(
        id: UUID = UUID(),
        raw: String,
        normalized: String,
        courtToken: String? = nil,
        jurisdiction: JurisdictionHint? = nil,
        isSupportedByLaunchScope: Bool
    ) {
        self.id = id
        self.raw = raw
        self.normalized = normalized
        self.courtToken = courtToken
        self.jurisdiction = jurisdiction
        self.isSupportedByLaunchScope = isSupportedByLaunchScope
    }
}

public struct CitationResolution: Identifiable, Hashable, Codable, Sendable {
    public let id: UUID
    public let citation: String
    public let normalized: String
    public let courtToken: String?
    public let state: CitationState
    public let source: String?
    public let sourceURL: URL?
    public let retrievedAt: Date?
    public let reason: String

    public init(
        id: UUID = UUID(),
        citation: String,
        normalized: String,
        courtToken: String? = nil,
        state: CitationState,
        source: String? = nil,
        sourceURL: URL? = nil,
        retrievedAt: Date? = nil,
        reason: String
    ) {
        self.id = id
        self.citation = citation
        self.normalized = normalized
        self.courtToken = courtToken
        self.state = state
        self.source = source
        self.sourceURL = sourceURL
        self.retrievedAt = retrievedAt
        self.reason = reason
    }
}

public struct ScanSummary: Equatable, Sendable {
    public let total: Int
    public let confirmed: Int
    public let couldNotConfirm: Int
    public let unsupported: Int

    public init(results: [CitationResolution]) {
        total = results.count
        confirmed = results.filter { $0.state == .confirmed }.count
        couldNotConfirm = results.filter { $0.state == .couldNotConfirm }.count
        unsupported = results.filter { $0.state == .unsupported }.count
    }

    public var hasStopAndCheckItems: Bool {
        couldNotConfirm > 0
    }
}

public struct CoverageSource: Identifiable, Hashable, Codable, Sendable {
    public var id: String { source }
    public let jurisdiction: String
    public let source: String
    public let completenessNote: String
    public let lastSnapshotDate: String?

    public init(jurisdiction: String, source: String, completenessNote: String, lastSnapshotDate: String? = nil) {
        self.jurisdiction = jurisdiction
        self.source = source
        self.completenessNote = completenessNote
        self.lastSnapshotDate = lastSnapshotDate
    }
}
