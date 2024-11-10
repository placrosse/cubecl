use std::{sync::Arc, thread};

use cubecl_common::benchmark::TimestampsResult;

use super::ComputeChannel;
use crate::{
    memory_management::MemoryUsage,
    server::{Binding, ComputeServer, CubeCount, Handle},
    storage::BindingResource,
    ExecutionMode,
};

/// Create a channel using a [multi-producer, single-consumer channel to communicate with
/// the compute server spawn on its own thread.
#[derive(Debug)]
pub struct MpscComputeChannel<Server>
where
    Server: ComputeServer,
{
    state: Arc<MpscComputeChannelState<Server>>,
}

#[derive(Debug)]
struct MpscComputeChannelState<Server>
where
    Server: ComputeServer,
{
    _handle: thread::JoinHandle<()>,
    sender: async_channel::Sender<Message<Server>>,
}

type Callback<Response> = async_channel::Sender<Response>;

enum Message<Server>
where
    Server: ComputeServer,
{
    Read(Binding, Callback<Vec<u8>>),
    GetResource(Binding, Callback<BindingResource<Server>>),
    Create(Vec<u8>, Callback<Handle>),
    Empty(usize, Callback<Handle>),
    ExecuteKernel((Server::Kernel, CubeCount, ExecutionMode), Vec<Binding>),
    Flush,
    SyncElapsed(Callback<TimestampsResult>),
    Sync(Callback<()>),
    GetMemoryUsage(Callback<MemoryUsage>),
    EnableTimestamps,
    DisableTimestamps,
}

impl<Server> MpscComputeChannel<Server>
where
    Server: ComputeServer + 'static,
{
    /// Create a new mpsc compute channel.
    pub fn new(mut server: Server) -> Self {
        let (sender, receiver) = async_channel::bounded(1);

        let _handle = thread::spawn(move || {
            // Run the whole procedure as one blocking future. This is much simpler than trying
            // to use some multithreaded executor.
            cubecl_common::future::block_on(async {
                while let Ok(message) = receiver.recv().await {
                    match message {
                        Message::Read(binding, callback) => {
                            let data = server.read(binding).await;
                            callback.send(data).await.unwrap();
                        }
                        Message::GetResource(binding, callback) => {
                            let data = server.get_resource(binding);
                            callback.send(data).await.unwrap();
                        }
                        Message::Create(data, callback) => {
                            let handle = server.create(&data);
                            callback.send(handle).await.unwrap();
                        }
                        Message::Empty(size, callback) => {
                            let handle = server.empty(size);
                            callback.send(handle).await.unwrap();
                        }
                        Message::ExecuteKernel(kernel, bindings) => unsafe {
                            server.execute(kernel.0, kernel.1, bindings, kernel.2);
                        },
                        Message::SyncElapsed(callback) => {
                            let duration = server.sync_elapsed().await;
                            callback.send(duration).await.unwrap();
                        }
                        Message::Sync(callback) => {
                            server.sync().await;
                            callback.send(()).await.unwrap();
                        }
                        Message::Flush => {
                            server.flush();
                        }
                        Message::GetMemoryUsage(callback) => {
                            callback.send(server.memory_usage()).await.unwrap();
                        }
                        Message::EnableTimestamps => {
                            server.enable_timestamps();
                        }
                        Message::DisableTimestamps => {
                            server.disable_timestamps();
                        }
                    };
                }
            });
        });

        let state = Arc::new(MpscComputeChannelState { sender, _handle });

        Self { state }
    }
}

impl<Server: ComputeServer> Clone for MpscComputeChannel<Server> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<Server> ComputeChannel<Server> for MpscComputeChannel<Server>
where
    Server: ComputeServer + 'static,
{
    async fn read(&self, binding: Binding) -> Vec<u8> {
        let sender = self.state.sender.clone();
        let (callback, response) = async_channel::unbounded();
        sender.send(Message::Read(binding, callback)).await.unwrap();
        handle_response(response.recv().await)
    }

    fn get_resource(&self, binding: Binding) -> BindingResource<Server> {
        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::GetResource(binding, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn create(&self, data: &[u8]) -> Handle {
        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::Create(data.to_vec(), callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn empty(&self, size: usize) -> Handle {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send_blocking(Message::Empty(size, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    unsafe fn execute(
        &self,
        kernel: Server::Kernel,
        count: CubeCount,
        bindings: Vec<Binding>,
        kind: ExecutionMode,
    ) {
        self.state
            .sender
            .send_blocking(Message::ExecuteKernel((kernel, count, kind), bindings))
            .unwrap()
    }

    fn flush(&self) {
        self.state.sender.send_blocking(Message::Flush).unwrap()
    }

    async fn sync(&self) {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send(Message::Sync(callback))
            .await
            .unwrap();
        handle_response(response.recv().await)
    }

    async fn sync_elapsed(&self) -> TimestampsResult {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send(Message::SyncElapsed(callback))
            .await
            .unwrap();
        handle_response(response.recv().await)
    }

    fn memory_usage(&self) -> crate::memory_management::MemoryUsage {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send_blocking(Message::GetMemoryUsage(callback))
            .unwrap();
        handle_response(response.recv_blocking())
    }

    fn enable_timestamps(&self) {
        self.state
            .sender
            .send_blocking(Message::EnableTimestamps)
            .unwrap();
    }

    fn disable_timestamps(&self) {
        self.state
            .sender
            .send_blocking(Message::DisableTimestamps)
            .unwrap();
    }
}

fn handle_response<Response, Err: core::fmt::Debug>(response: Result<Response, Err>) -> Response {
    match response {
        Ok(val) => val,
        Err(err) => panic!("Can't connect to the server correctly {err:?}"),
    }
}
