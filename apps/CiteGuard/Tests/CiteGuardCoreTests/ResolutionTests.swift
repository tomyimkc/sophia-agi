import Foundation
import XCTest
@testable import CiteGuardCore

final class ResolutionTests: XCTestCase {
    func testOfflineSnapshotConfirmsOnlyRegisterHits() async {
        let snapshot = BundledAuthoritySnapshot()
        let extractor = HKCitationExtractor()
        let citations = extractor.extract(from: "See [2025] HKCFI 808 and Wong v Lee [2025] HKCFI 9999.")

        let results = citations.map { snapshot.resolve($0) }

        XCTAssertEqual(results.map(\.state), [.confirmed, .couldNotConfirm])
        XCTAssertTrue(results[1].reason.localizedCaseInsensitiveContains("stop and check"))
    }

    func testUnsupportedCitationIsNeutralOutOfScope() async {
        let snapshot = BundledAuthoritySnapshot()
        let extractor = HKCitationExtractor()
        let citation = extractor.extract(from: "See Varghese v. China Southern Airlines, 925 F.3d 1339.").first!

        let result = snapshot.resolve(citation)

        XCTAssertEqual(result.state, .unsupported)
        XCTAssertTrue(result.reason.localizedCaseInsensitiveContains("not checked"))
    }

    func testMissingAPIKeyFailsClosedForLiveMode() async {
        let resolver = CiteGuardAPIClient(apiKey: nil)
        let extractor = HKCitationExtractor()
        let candidates = extractor.extract(from: "Per [2025] HKCFI 808.")

        let results = await resolver.resolve(candidates: candidates, mode: .live, jurisdictionHint: .hk)

        XCTAssertEqual(results.first?.state, .couldNotConfirm)
        XCTAssertTrue(results.first?.reason.localizedCaseInsensitiveContains("could not") == true)
    }
}
