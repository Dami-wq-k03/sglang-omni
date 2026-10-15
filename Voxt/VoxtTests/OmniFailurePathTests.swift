// OmniFailurePathTests.swift
// Failure paths of the Omni client that need no model: live socket errors,
// orphaned runtimes, retirement racing a launch, and what errors may say.

import Network
import XCTest
@testable import Voxt

/// A loopback WebSocket server with one scripted event per connection.
private final class LiveEventServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "omni-error-event-server")
    private var connections: [NWConnection] = []
    let ready = DispatchSemaphore(value: 0)

    init(events: [String]) throws {
        precondition(!events.isEmpty)
        let parameters = NWParameters.tcp
        parameters.defaultProtocolStack.applicationProtocols.insert(NWProtocolWebSocket.Options(), at: 0)
        listener = try NWListener(using: parameters, on: .any)
        listener.stateUpdateHandler = { [ready] state in
            if case .ready = state { ready.signal() }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            self.connections.append(connection)
            let event = events[min(self.connections.count - 1, events.count - 1)]
            connection.start(queue: self.queue)
            connection.receiveMessage { _, _, _, _ in
                let metadata = NWProtocolWebSocket.Metadata(opcode: .text)
                let context = NWConnection.ContentContext(identifier: "event", metadata: [metadata])
                connection.send(
                    content: Data(event.utf8),
                    contentContext: context,
                    isComplete: true,
                    completion: .contentProcessed { _ in }
                )
            }
        }
        listener.start(queue: queue)
    }

    var port: Int { Int(listener.port?.rawValue ?? 0) }
    var connectionCount: Int { queue.sync { connections.count } }

    func stop() {
        connections.forEach { $0.cancel() }
        listener.cancel()
    }
}

final class OmniFailurePathTests: XCTestCase {
    @MainActor
    func testLiveSetupFailureIsRetriedAndReportedDuringRecording() async {
        let attempts = StartupAttemptCounter()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await attempts.increment()
            throw OmniASRRuntimeError.launchFailed("injected startup failure")
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        var failures: [String] = []
        transcriber.onRuntimeFailure = { failures.append($0) }
        transcriber.isRecording = true
        transcriber.isModelInitializing = true

        await transcriber.startNativeLiveSession(revision: 0).value

        let attemptCount = await attempts.value
        XCTAssertEqual(attemptCount, 2)
        XCTAssertNotNil(transcriber.consumePendingRuntimeFailureMessage())
        XCTAssertEqual(failures.count, 1)
        XCTAssertFalse(transcriber.isModelInitializing)
        XCTAssertFalse(manager.hasActiveUse)
        transcriber.isRecording = false
        await manager.shutdownForApplicationTermination()
    }

    @MainActor
    func testTransientLoadFailureRecoversLivePreviewWithoutRestarting() async throws {
        try await verifyLiveRecovery(failFirstLoad: true)
    }

    @MainActor
    func testTransientConnectionFailureRecoversLivePreviewWithoutRestarting() async throws {
        try await verifyLiveRecovery(failFirstLoad: false)
    }

    @MainActor
    func testRepeatedConnectionFailureIsReportedAndStopsRetrying() async throws {
        try await verifyLiveRecovery(failFirstLoad: false, recoverySucceeds: false)
    }

    @MainActor
    func testConnectionFailureAfterLoadRetryDoesNotRetryAgain() async throws {
        try await verifyLiveRecovery(failFirstLoad: true, recoverySucceeds: false)
    }

    @MainActor
    private func verifyLiveRecovery(failFirstLoad: Bool, recoverySucceeds: Bool = true) async throws {
        let previewEvent = """
        {"type":"transcription.segment","segment_id":0,"text":"recovered preview","is_final":false,"event_index":1}
        """
        let failureEvent = """
        {"type":"error","error":{"code":"injected_connection_failure"}}
        """
        let finalEvent = recoverySucceeds ? previewEvent : failureEvent
        let server = try LiveEventServer(events: failFirstLoad ? [finalEvent] : [failureEvent, finalEvent])
        defer { server.stop() }
        XCTAssertEqual(server.ready.wait(timeout: .now() + 5), .success)
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-startup-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        let executable = scratch.appendingPathComponent("server")
        try """
        #!/bin/sh
        echo '{"event":"ready","host":"127.0.0.1","port":\(server.port),"model_name":"m","server_pid":'"$$"'}'
        read -r command
        """.write(to: executable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        let runtime = OmniASRRuntime(
            kind: .qwen3ASR,
            modelDirectory: scratch,
            configuration: OmniBackendConfiguration(runtimeExecutable: executable)
        )
        let attempts = StartupAttemptCounter()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await attempts.increment()
            if failFirstLoad, await attempts.value == 1 {
                throw OmniASRRuntimeError.launchFailed("injected transient failure")
            }
            _ = try await runtime.prepare()
            return MLXLoadedModelBox(loaded: .omni(runtime))
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        let result = expectation(description: "live startup completed")
        transcriber.onPartialTranscription = { text in
            XCTAssertTrue(recoverySucceeds)
            XCTAssertEqual(text, "recovered preview")
            result.fulfill()
        }
        transcriber.onRuntimeFailure = { _ in
            XCTAssertFalse(recoverySucceeds)
            result.fulfill()
        }
        transcriber.isRecording = true

        await transcriber.startNativeLiveSession(revision: 0).value
        await fulfillment(of: [result], timeout: 5)

        let attemptCount = await attempts.value
        XCTAssertEqual(attemptCount, failFirstLoad ? 2 : 1)
        XCTAssertEqual(server.connectionCount, failFirstLoad ? 1 : 2)
        if recoverySucceeds {
            XCTAssertEqual(transcriber.transcribedText, "recovered preview")
            XCTAssertNil(transcriber.consumePendingRuntimeFailureMessage())
        } else {
            XCTAssertTrue(transcriber.transcribedText.isEmpty)
            XCTAssertNotNil(transcriber.consumePendingRuntimeFailureMessage())
        }
        transcriber.isRecording = false
        transcriber.releaseNativeLiveSession(cancelSession: true)
        await manager.shutdownForApplicationTermination()
        await runtime.retire()
    }

    @MainActor
    func testStoppingDuringLiveSetupDoesNotRetryOrReportStaleFailure() async {
        let barrier = ManualTaskBarrier()
        let attempts = StartupAttemptCounter()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await attempts.increment()
            await barrier.wait()
            throw OmniASRRuntimeError.launchFailed("injected stale failure")
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        transcriber.onRuntimeFailure = { _ in XCTFail("Stopped setup reported a stale failure") }
        transcriber.isRecording = true
        let setup = transcriber.startNativeLiveSession(revision: 0)
        await barrier.waitUntilEntered()
        transcriber.isRecording = false
        barrier.release()
        await setup.value

        let attemptCount = await attempts.value
        XCTAssertEqual(attemptCount, 1)
        XCTAssertNil(transcriber.consumePendingRuntimeFailureMessage())
        XCTAssertFalse(manager.hasActiveUse)
        await manager.shutdownForApplicationTermination()
    }

    @MainActor
    func testCancelledLiveSetupDoesNotRetryOrReportFailure() async {
        let barrier = ManualTaskBarrier()
        let attempts = StartupAttemptCounter()
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo) { _ in
            await attempts.increment()
            await barrier.wait()
            throw OmniASRRuntimeError.launchFailed("injected cancelled failure")
        }
        let transcriber = MLXTranscriber(modelManager: manager)
        transcriber.onRuntimeFailure = { _ in XCTFail("Cancelled setup reported a failure") }
        transcriber.isRecording = true
        let setup = transcriber.startNativeLiveSession(revision: 0)
        await barrier.waitUntilEntered()
        setup.cancel()
        barrier.release()
        await setup.value

        let attemptCount = await attempts.value
        XCTAssertEqual(attemptCount, 1)
        XCTAssertNil(transcriber.consumePendingRuntimeFailureMessage())
        XCTAssertFalse(manager.hasActiveUse)
        transcriber.isRecording = false
        await manager.shutdownForApplicationTermination()
    }

    /// A server error ends the live session: the stream finishes, so whoever
    /// forwards it releases its runtime lease, and the message carries no text
    /// the server may have echoed from the request.
    func testALiveSocketErrorEventEndsTheSession() async throws {
        let server = try LiveEventServer(events: ["""
        {"type": "error", "error": {"type": "invalid_request_error", "code": "invalid_event", "message": "input_value='U0VDUkVUIEFVRElP'"}}
        """])
        defer { server.stop() }
        XCTAssertEqual(server.ready.wait(timeout: .now() + 5), .success)
        let session = OmniRealtimeTranscriptionSession(
            endpoint: OmniServerEndpoint(host: "127.0.0.1", port: server.port, modelName: "m", serverProcessIdentifier: 0),
            language: nil
        )
        session.feedAudio(samples: [0, 0.1, 0.2])

        let drained = Task { () -> [OmniLiveEvent] in
            var received: [OmniLiveEvent] = []
            for await event in session.events { received.append(event) }
            return received
        }
        let finished = Task { () -> Bool in
            try? await Task.sleep(for: .seconds(5))
            return false
        }
        let events = await withTaskGroup(of: [OmniLiveEvent]?.self) { group in
            group.addTask { await drained.value }
            group.addTask { _ = await finished.value; return nil }
            let first = await group.next() ?? nil
            group.cancelAll()
            drained.cancel()
            return first
        }
        finished.cancel()

        let received = try XCTUnwrap(events, "the live session never ended after a server error")
        guard case .failed(let message) = received.last else {
            return XCTFail("expected a failure event, got \(received)")
        }
        XCTAssertFalse(message.contains("U0VDUkVUIEFVRElP"))
        XCTAssertTrue(message.contains("invalid_event"))
    }

    func testErrorDescriptionsCarryNoRequestOrTranscriptText() {
        let secret = "dictionary term and transcript text"
        let errors: [OmniTranscriptionError] = [
            .httpStatus(422, "{\"detail\": [{\"input\": \"\(secret)\"}]}"),
            .malformedEvent("{\"type\": \"transcript.text.done\", \"text\": \"\(secret)\"}"),
        ]
        for error in errors {
            let description = error.errorDescription ?? ""
            XCTAssertFalse(description.contains(secret), description)
        }
        XCTAssertTrue(OmniTranscriptionError.httpStatus(422, secret).errorDescription?.contains("422") ?? false)
    }

    @MainActor
    func testTheLedgerReleasesRuntimesNoLoadWillAdopt() async {
        let scratch = FileManager.default.temporaryDirectory
        let configuration = OmniBackendConfiguration(
            runtimeExecutable: URL(fileURLWithPath: "/usr/bin/false")
        )
        let adopted = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch, configuration: configuration)
        let orphan = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch, configuration: configuration)
        let ledger = OmniRuntimeLedger()
        ledger.track(adopted)
        ledger.adopt(adopted)
        ledger.track(orphan)

        ledger.releaseUnadopted()
        await ledger.waitForRetirements()

        XCTAssertTrue(ledger.pendingRuntimes().isEmpty)
        let orphanState = await orphan.state
        let adoptedState = await adopted.state
        XCTAssertEqual(orphanState, .stopped)
        XCTAssertEqual(adoptedState, .idle)
    }

    /// Retiring a runtime whose launch has not started yet must keep the
    /// server process from starting at all. A smoke check: the window between
    /// prepare() queueing the launch and the launch running cannot be hit on
    /// demand, so this passes without the guard too; it catches leftovers.
    func testRetiringBeforeTheLaunchRunsStartsNoServerProcess() async throws {
        let token = "voxt-omni-retire-race-\(UUID().uuidString)"
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(token, isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        let script = scratch.appendingPathComponent("qwen3_asr_server")
        // Note (Jiaxin Deng): stands in for the runtime; exits once Voxt writes or closes its control pipe.
        try "#!/bin/sh\nexec /bin/sh -c 'read line' \(token)\n".write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let configuration = OmniBackendConfiguration(
            runtimeExecutable: script,
            startupTimeoutSeconds: 5
        )

        var preparations: [Task<OmniServerEndpoint, Error>] = []
        for _ in 0..<20 {
            let runtime = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch, configuration: configuration)
            preparations.append(Task { try await runtime.prepare() })
            await runtime.retire()
        }
        try await Task.sleep(for: .milliseconds(500))
        let leftover = Self.processes(matching: token)
        // Note (Jiaxin Deng): a server started after retirement would wait on its control pipe forever.
        Self.kill(matching: token)
        preparations.forEach { $0.cancel() }

        XCTAssertEqual(leftover, 0, "a server process started after its runtime retired")
    }

    private static func kill(matching token: String) {
        let pkill = Process()
        pkill.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        pkill.arguments = ["-f", token]
        try? pkill.run()
        pkill.waitUntilExit()
    }

    func testQwenEnergyCutsMayPassTheChunkEndUnlessThatWouldExceedTheServerLimit() {
        XCTAssertTrue(OmniASRRuntime.qwenCutMayPassChunkEnd(chunkDurationSeconds: 90))
        XCTAssertTrue(OmniASRRuntime.qwenCutMayPassChunkEnd(chunkDurationSeconds: 1195))
        XCTAssertFalse(OmniASRRuntime.qwenCutMayPassChunkEnd(chunkDurationSeconds: 1196))
        XCTAssertFalse(OmniASRRuntime.qwenCutMayPassChunkEnd(chunkDurationSeconds: 1200))
    }

    private static func processes(matching token: String) -> Int {
        let pgrep = Process()
        pgrep.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        pgrep.arguments = ["-f", token]
        let output = Pipe()
        pgrep.standardOutput = output
        try? pgrep.run()
        pgrep.waitUntilExit()
        let text = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        return text.split(whereSeparator: \.isNewline).count
    }
}

private actor StartupAttemptCounter {
    private(set) var value = 0

    func increment() {
        value += 1
    }
}
