import Foundation

public protocol CitationExtracting: Sendable {
    func extract(from text: String) -> [CitationCandidate]
}

public struct HKCitationExtractor: CitationExtracting {
    private static let hkCourts: Set<String> = [
        "HKCFA", "HKCA", "HKCFI", "HKDC", "HKFC", "HKLDT", "HKCT", "HKLT", "HKMAGC", "HKCRC"
    ]

    private let neutralPattern = #"\[\s*(\d{4})\s*\]\s+([A-Za-z]{2,8})\s+(\d+)(\s*\([A-Za-z]+\))?"#
    private let capPattern = #"\bCap\.?\s*(\d+[A-Za-z]?)\b"#
    private let usReporterPattern = #"\b(\d+)\s+(U\.?\s?S\.?|S\.?\s?Ct\.?|L\.?\s?Ed\.?(?:\s?2d)?|F\.?\s?Supp\.?(?:\s?[23]d)?|F\.?\s?(?:2d|3d|4th)?)\s+(\d+)\b"#

    public init() {}

    public func extract(from text: String) -> [CitationCandidate] {
        var candidates: [CitationCandidate] = []
        candidates.append(contentsOf: extractNeutralCitations(from: text))
        candidates.append(contentsOf: extractCapReferences(from: text))
        candidates.append(contentsOf: extractUSReporterCitations(from: text))
        return deduplicated(candidates)
    }

    public func normalize(_ citation: String) -> String {
        let collapsed = citation.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if let match = firstMatch(pattern: neutralPattern, in: collapsed) {
            let year = match.group(1, in: collapsed)
            let court = match.group(2, in: collapsed).uppercased()
            let number = match.group(3, in: collapsed)
            let division = match.group(4, in: collapsed)
                .replacingOccurrences(of: #"\s+"#, with: "", options: .regularExpression)
            return "[\(year)] \(court) \(number)\(division)"
        }

        if let match = firstMatch(pattern: capPattern, in: collapsed, options: [.caseInsensitive]) {
            return "Cap. \(match.group(1, in: collapsed).uppercased())"
        }

        if let match = firstMatch(pattern: usReporterPattern, in: collapsed) {
            let reporter = canonicalUSReporter(match.group(2, in: collapsed)) ?? match.group(2, in: collapsed)
            return "\(match.group(1, in: collapsed)) \(reporter) \(match.group(3, in: collapsed))"
        }

        return collapsed
    }

    private func extractNeutralCitations(from text: String) -> [CitationCandidate] {
        matches(pattern: neutralPattern, in: text).map { match in
            let raw = match.group(0, in: text)
            let court = match.group(2, in: text).uppercased()
            let normalized = normalize(raw)
            let isHK = Self.hkCourts.contains(court)
            return CitationCandidate(
                raw: raw,
                normalized: normalized,
                courtToken: court,
                jurisdiction: isHK ? .hk : inferredNeutralJurisdiction(court),
                isSupportedByLaunchScope: isHK
            )
        }
    }

    private func extractCapReferences(from text: String) -> [CitationCandidate] {
        matches(pattern: capPattern, in: text, options: [.caseInsensitive]).map { match in
            let raw = match.group(0, in: text)
            return CitationCandidate(
                raw: raw,
                normalized: normalize(raw),
                courtToken: nil,
                jurisdiction: .hk,
                isSupportedByLaunchScope: true
            )
        }
    }

    private func extractUSReporterCitations(from text: String) -> [CitationCandidate] {
        matches(pattern: usReporterPattern, in: text).compactMap { match in
            let raw = match.group(0, in: text)
            guard canonicalUSReporter(match.group(2, in: text)) != nil else {
                return nil
            }
            return CitationCandidate(
                raw: raw,
                normalized: normalize(raw),
                courtToken: nil,
                jurisdiction: .us,
                isSupportedByLaunchScope: false
            )
        }
    }

    private func inferredNeutralJurisdiction(_ court: String) -> JurisdictionHint? {
        let ukCourts: Set<String> = ["EWHC", "EWCA", "UKSC", "UKHL", "UKPC", "EWFC", "EWCOP", "UKUT", "UKEAT"]
        if ukCourts.contains(court) {
            return .uk
        }
        return nil
    }

    private func canonicalUSReporter(_ reporter: String) -> String? {
        let key = reporter.replacingOccurrences(of: #"\s+"#, with: "", options: .regularExpression).lowercased()
        let reporters: [String: String] = [
            "u.s.": "U.S.", "us": "U.S.",
            "s.ct.": "S. Ct.", "sct": "S. Ct.", "s.ct": "S. Ct.",
            "l.ed.": "L. Ed.", "l.ed.2d": "L. Ed. 2d", "led": "L. Ed.", "led2d": "L. Ed. 2d",
            "f.": "F.", "f": "F.", "f.2d": "F.2d", "f2d": "F.2d",
            "f.3d": "F.3d", "f3d": "F.3d", "f.4th": "F.4th", "f4th": "F.4th",
            "f.supp.": "F. Supp.", "fsupp": "F. Supp.", "f.supp": "F. Supp.",
            "f.supp.2d": "F. Supp. 2d", "fsupp2d": "F. Supp. 2d",
            "f.supp.3d": "F. Supp. 3d", "fsupp3d": "F. Supp. 3d"
        ]
        return reporters[key]
    }

    private func deduplicated(_ candidates: [CitationCandidate]) -> [CitationCandidate] {
        var seen = Set<String>()
        var ordered: [CitationCandidate] = []
        for candidate in candidates where !seen.contains(candidate.normalized) {
            seen.insert(candidate.normalized)
            ordered.append(candidate)
        }
        return ordered
    }

    private func firstMatch(pattern: String, in text: String, options: NSRegularExpression.Options = []) -> NSTextCheckingResult? {
        matches(pattern: pattern, in: text, options: options).first
    }

    private func matches(pattern: String, in text: String, options: NSRegularExpression.Options = []) -> [NSTextCheckingResult] {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
            return []
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, range: range)
    }
}

private extension NSTextCheckingResult {
    func group(_ index: Int, in text: String) -> String {
        let range = self.range(at: index)
        guard range.location != NSNotFound, let swiftRange = Range(range, in: text) else {
            return ""
        }
        return String(text[swiftRange])
    }
}
