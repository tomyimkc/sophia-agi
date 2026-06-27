import CiteGuardCore
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var model = CiteGuardViewModel()
    @State private var importing = false

    var body: some View {
        NavigationStack {
            List {
                disclaimerSection
                intakeSection
                summarySection
                if !model.results.isEmpty {
                    resultsSection
                }
                coverageSection
                privacySection
            }
            .navigationTitle("CiteGuard")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await model.scanPastedText() }
                    } label: {
                        Label("Scan", systemImage: "text.magnifyingglass")
                    }
                    .disabled(model.isScanning)
                }
            }
            .fileImporter(
                isPresented: $importing,
                allowedContentTypes: [.pdf, .plainText, .text, .item],
                allowsMultipleSelection: false
            ) { result in
                guard case .success(let urls) = result, let url = urls.first else { return }
                Task { await model.importDocument(from: url) }
            }
            .task {
                await model.refreshCoverage()
            }
        }
    }

    private var disclaimerSection: some View {
        Section {
            Text("Not legal advice. Confirms citation EXISTENCE only. Does not verify that an authority supports your proposition or is current law. You remain responsible for verifying every citation.")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.primary)
            Text("唔係法律意見。只檢查引文是否可在來源找到；不判斷案例是否支持你的論點，亦不確認是否仍然有效。你仍須自行核對每條引文。")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    private var intakeSection: some View {
        Section("Document") {
            Picker("Mode", selection: $model.mode) {
                Text("Offline snapshot").tag(ResolutionMode.cache)
                Text("Live sources").tag(ResolutionMode.live)
            }
            .pickerStyle(.segmented)

            if model.mode == .cache {
                Label("Snapshot-limited: missing results may be outside or newer than bundled coverage.", systemImage: "wifi.slash")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            TextEditor(text: $model.documentText)
                .frame(minHeight: 160)
                .accessibilityLabel("Paste legal document text")

            HStack {
                Button {
                    importing = true
                } label: {
                    Label("Open File", systemImage: "doc")
                }

                Spacer()

                Button {
                    Task { await model.scanPastedText() }
                } label: {
                    if model.isScanning {
                        ProgressView()
                    } else {
                        Label("Check Citations", systemImage: "exclamationmark.triangle")
                    }
                }
                .buttonStyle(.borderedProminent)
            }

            if let error = model.lastError {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.footnote)
            }
        }
    }

    private var summarySection: some View {
        Section("Summary") {
            let summary = model.summary
            HStack(spacing: 12) {
                SummaryTile(label: "Citations", value: summary.total, tint: .primary)
                SummaryTile(label: "Confirmed", value: summary.confirmed, tint: .blue)
                SummaryTile(label: "Could not confirm", value: summary.couldNotConfirm, tint: .orange)
                SummaryTile(label: "Unsupported", value: summary.unsupported, tint: .gray)
            }
            .accessibilityElement(children: .combine)

            if summary.hasStopAndCheckItems {
                Label("Stop and check every citation marked Could not confirm.", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .font(.headline)
            }
        }
    }

    private var resultsSection: some View {
        Section("Citation Results") {
            ForEach(model.results) { result in
                NavigationLink {
                    CitationDetailView(result: result)
                } label: {
                    CitationRow(result: result)
                }
            }
        }
    }

    private var coverageSection: some View {
        Section("Coverage") {
            ForEach(model.coverage) { source in
                VStack(alignment: .leading, spacing: 4) {
                    Text(source.source)
                        .font(.headline)
                    Text(source.completenessNote)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var privacySection: some View {
        Section("What Leaves This Device") {
            Text("Document text is parsed on-device. In live mode, only extracted citation strings, capped at 120 characters each, are sent to the resolution API.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            if model.privacyEntries.isEmpty {
                Text("No live requests recorded.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(model.privacyEntries.prefix(5)) { entry in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(entry.createdAt, style: .date)
                            .font(.caption.weight(.semibold))
                        Text(entry.citationsSent.joined(separator: ", "))
                            .font(.caption)
                    }
                }
            }
        }
    }
}

private struct SummaryTile: View {
    let label: String
    let value: Int
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(value)")
                .font(.title3.weight(.bold))
                .foregroundStyle(tint)
                .minimumScaleFactor(0.7)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, minHeight: 62, alignment: .leading)
    }
}

private struct CitationRow: View {
    let result: CitationResolution

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(result.normalized)
                    .font(.headline)
                Spacer()
                StateBadge(state: result.state, source: result.source)
            }
            Text(result.reason)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .lineLimit(3)
        }
        .padding(.vertical, 4)
    }
}

private struct StateBadge: View {
    let state: CitationState
    var source: String? = nil

    var body: some View {
        Label(title, systemImage: icon)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .foregroundStyle(tint)
            .background(tint.opacity(0.12), in: Capsule())
    }

    private var title: String {
        switch state {
        case .confirmed:
            return "Confirmed in \(source ?? "source")"
        case .couldNotConfirm:
            return "Could not confirm"
        case .unsupported:
            return "Unsupported"
        }
    }

    private var tint: Color {
        switch state {
        case .confirmed:
            return .blue
        case .couldNotConfirm:
            return .orange
        case .unsupported:
            return .gray
        }
    }

    private var icon: String {
        switch state {
        case .confirmed:
            return "doc.text.magnifyingglass"
        case .couldNotConfirm:
            return "exclamationmark.triangle.fill"
        case .unsupported:
            return "minus.circle"
        }
    }
}

private struct CitationDetailView: View {
    let result: CitationResolution

    var body: some View {
        List {
            Section("Citation") {
                Text(result.normalized)
                StateBadge(state: result.state, source: result.source)
            }

            Section("Reason") {
                Text(result.reason)
                if result.state == .couldNotConfirm {
                    Text("This does not prove the citation is fake. It may be outside, newer than, or ambiguous within current coverage.")
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(.orange)
                }
            }

            Section("Source") {
                Text(result.source ?? "No source confirmed this citation.")
                if let retrievedAt = result.retrievedAt {
                    Text("Retrieved \(retrievedAt.formatted(date: .abbreviated, time: .shortened))")
                }
                if let url = result.sourceURL {
                    Link("Open Source", destination: url)
                }
            }
        }
        .navigationTitle("Citation")
    }
}
