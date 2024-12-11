use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::shared::CompilationOptions;

use crate::runtime::HipCompiler;

use super::fence::{Fence, SyncStream};
use super::storage::HipStorage;
use super::{uninit_vec, HipResource};
use cubecl_core::compute::DebugInformation;
use cubecl_core::ir::CubeDim;
use cubecl_core::Feature;
use cubecl_core::{prelude::*, KernelId};
use cubecl_hip_sys::{hiprtcResult_HIPRTC_SUCCESS, HIP_SUCCESS};
use cubecl_runtime::debug::{DebugLogger, ProfileLevel};
use cubecl_runtime::memory_management::MemoryUsage;
use cubecl_runtime::storage::BindingResource;
use cubecl_runtime::{
    memory_management::MemoryManagement,
    server::{self, ComputeServer},
};
use cubecl_runtime::{ExecutionMode, TimestampsError, TimestampsResult};
use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::future::Future;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug)]
pub struct HipServer {
    ctx: HipContext,
    logger: DebugLogger,
}

#[derive(Debug)]
pub(crate) struct HipContext {
    context: cubecl_hip_sys::hipCtx_t,
    stream: cubecl_hip_sys::hipStream_t,
    memory_management: MemoryManagement<HipStorage>,
    module_names: HashMap<KernelId, HipCompiledKernel>,
    timestamps: KernelTimestamps,
    compilation_options: CompilationOptions,
}

#[derive(Debug)]
struct HipCompiledKernel {
    _module: cubecl_hip_sys::hipModule_t,
    func: cubecl_hip_sys::hipFunction_t,
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
}

#[derive(Debug)]
enum KernelTimestamps {
    Inferred { start_time: Instant },
    Disabled,
}

impl KernelTimestamps {
    fn enable(&mut self) {
        if !matches!(self, Self::Disabled) {
            return;
        }

        *self = Self::Inferred {
            start_time: Instant::now(),
        };
    }

    fn disable(&mut self) {
        *self = Self::Disabled;
    }
}

unsafe impl Send for HipServer {}

impl HipServer {
    fn read_sync(&mut self, binding: server::Binding) -> Vec<u8> {
        let ctx = self.get_context();
        let resource = ctx.memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        );

        let mut data = uninit_vec(resource.size as usize);
        unsafe {
            let status = cubecl_hip_sys::hipMemcpyDtoHAsync(
                data.as_mut_ptr() as *mut _,
                resource.ptr,
                resource.size as usize,
                ctx.stream,
            );
            assert_eq!(status, HIP_SUCCESS, "Should copy data from device to host");
        };
        ctx.sync();
        data
    }

    fn read_async(
        &mut self,
        bindings: Vec<server::Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + 'static + Send {
        let ctx = self.get_context();
        let mut result = Vec::with_capacity(bindings.len());

        for binding in bindings {
            let resource = ctx.memory_management.get_resource(
                binding.memory,
                binding.offset_start,
                binding.offset_end,
            );

            let mut data = uninit_vec(resource.size as usize);
            unsafe {
                let status = cubecl_hip_sys::hipMemcpyDtoHAsync(
                    data.as_mut_ptr() as *mut _,
                    resource.ptr,
                    resource.size as usize,
                    ctx.stream,
                );
                assert_eq!(status, HIP_SUCCESS, "Should copy data from device to host");
            };
            result.push(data);
        }

        let fence = ctx.fence();

        async move {
            fence.wait();
            result
        }
    }

    fn sync_stream_async(&mut self) -> impl Future<Output = ()> + 'static + Send {
        let ctx = self.get_context();
        // We can't use a fence here because no action has been recorded on the context.
        // We need at least one action to be recorded after the context is initialized
        // with `cudarc::driver::result::ctx::set_current(self.ctx.context)` for the fence
        // to have any effect. Otherwise, it seems to be ignored.
        let sync = ctx.lazy_sync_stream();

        async move {
            sync.wait();
        }
    }
}

impl ComputeServer for HipServer {
    type Kernel = Box<dyn CubeTask<HipCompiler>>;
    type Storage = HipStorage;
    type Feature = Feature;

    fn read(
        &mut self,
        bindings: Vec<server::Binding>,
    ) -> impl Future<Output = Vec<Vec<u8>>> + 'static {
        self.read_async(bindings)
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.ctx.memory_usage()
    }

    fn create(&mut self, data: &[u8]) -> server::Handle {
        let handle = self.empty(data.len());
        let ctx = self.get_context();

        let binding = handle.clone().binding();
        let resource = ctx.memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        );

        unsafe {
            let status = cubecl_hip_sys::hipMemcpyHtoDAsync(
                resource.ptr,
                data as *const _ as *mut _,
                data.len(),
                ctx.stream,
            );
            assert_eq!(status, HIP_SUCCESS, "Should send data to device");
        }
        handle
    }

    fn empty(&mut self, size: usize) -> server::Handle {
        let ctx = self.get_context();
        let handle = ctx.memory_management.reserve(size as u64, None);
        server::Handle::new(handle, None, None, size as u64)
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Vec<server::Binding>,
        mode: ExecutionMode,
    ) {
        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        let profile_level = self.logger.profile_level();
        let profile_info = if profile_level.is_some() {
            Some((kernel.name(), kernel_id.clone()))
        } else {
            None
        };

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: CUDA doesn't have an exact equivalen of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                let data = self.read_sync(binding);
                let data = bytemuck::cast_slice(&data);
                assert!(
                    data.len() == 3,
                    "Dynamic cube count should contain 3 values"
                );
                (data[0], data[1], data[2])
            }
        };

        let (ctx, logger) = self.get_context_with_logger();

        if !ctx.module_names.contains_key(&kernel_id) {
            ctx.compile_kernel(&kernel_id, kernel, logger, mode);
        }

        let resources = bindings
            .into_iter()
            .map(|binding| {
                ctx.memory_management.get_resource(
                    binding.memory,
                    binding.offset_start,
                    binding.offset_end,
                )
            })
            .collect::<Vec<_>>();

        if let Some(level) = profile_level {
            let start = std::time::SystemTime::now();
            ctx.execute_task(kernel_id, count, resources);
            ctx.sync();

            let (name, kernel_id) = profile_info.unwrap();
            let info = match level {
                ProfileLevel::Basic | ProfileLevel::Medium => {
                    if let Some(val) = name.split("<").next() {
                        val.split("::").last().unwrap_or(name).to_string()
                    } else {
                        name.to_string()
                    }
                }
                cubecl_runtime::debug::ProfileLevel::Full => {
                    format!("{name}: {kernel_id} CubeCount {count:?}")
                }
            };

            self.logger
                .register_profiled(info, start.elapsed().unwrap());
        } else {
            ctx.execute_task(kernel_id, count, resources);
            ctx.sync();
        }
    }

    fn flush(&mut self) {}

    fn sync(&mut self) -> impl Future<Output = ()> + 'static {
        self.logger.profile_summary();
        self.sync_stream_async()
    }

    fn sync_elapsed(&mut self) -> impl Future<Output = TimestampsResult> + 'static {
        self.logger.profile_summary();

        let ctx = self.get_context();
        ctx.sync();

        let duration = match &mut ctx.timestamps {
            KernelTimestamps::Inferred { start_time } => {
                let duration = start_time.elapsed();
                *start_time = Instant::now();
                Ok(duration)
            }
            KernelTimestamps::Disabled => Err(TimestampsError::Disabled),
        };

        async move { duration }
    }

    fn get_resource(&mut self, binding: server::Binding) -> BindingResource<Self> {
        let ctx = self.get_context();
        BindingResource::new(
            binding.clone(),
            ctx.memory_management.get_resource(
                binding.memory,
                binding.offset_start,
                binding.offset_end,
            ),
        )
    }

    fn enable_timestamps(&mut self) {
        self.ctx.timestamps.enable();
    }

    fn disable_timestamps(&mut self) {
        if self.logger.profile_level().is_none() {
            self.ctx.timestamps.disable();
        }
    }
}

impl HipContext {
    pub fn new(
        memory_management: MemoryManagement<HipStorage>,
        compilation_options: CompilationOptions,
        stream: cubecl_hip_sys::hipStream_t,
        context: cubecl_hip_sys::hipCtx_t,
    ) -> Self {
        Self {
            memory_management,
            module_names: HashMap::new(),
            stream,
            context,
            timestamps: KernelTimestamps::Disabled,
            compilation_options,
        }
    }

    fn fence(&mut self) -> Fence {
        Fence::new(self.stream)
    }

    fn lazy_sync_stream(&mut self) -> SyncStream {
        SyncStream::new(self.stream)
    }

    fn sync(&mut self) {
        unsafe {
            let status = cubecl_hip_sys::hipStreamSynchronize(self.stream);
            assert_eq!(
                status, HIP_SUCCESS,
                "Should successfully synchronize stream"
            );
        };
        self.memory_management.storage().flush();
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.memory_management.memory_usage()
    }

    fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        cube_kernel: Box<dyn CubeTask<HipCompiler>>,
        logger: &mut DebugLogger,
        mode: ExecutionMode,
    ) {
        // CubeCL compilation
        // jitc = just-in-time compiled
        let mut jitc_kernel = cube_kernel.compile(&self.compilation_options, mode);
        let func_name = CString::new(jitc_kernel.entrypoint_name.clone()).unwrap();

        if logger.is_activated() {
            jitc_kernel.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&jitc_kernel.source) {
                jitc_kernel.source = formatted;
            }
        }
        let jitc_kernel = logger.debug(jitc_kernel);

        // Create HIP Program
        let program = unsafe {
            let source = CString::new(jitc_kernel.source.clone()).unwrap();
            let mut program: cubecl_hip_sys::hiprtcProgram = std::ptr::null_mut();
            let status = cubecl_hip_sys::hiprtcCreateProgram(
                &mut program,
                source.as_ptr(),
                std::ptr::null(), // program name seems unnecessary
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should create the program"
            );
            program
        };
        // Compile HIP program
        // options
        let include_path = include_path();
        let include_option = format!("-I{}", include_path.display());
        let include_option_cstr = CString::new(include_option).unwrap();
        // needed for rocWMMA extension to compile
        let cpp_std_option_cstr = CString::new("--std=c++17").unwrap();
        let mut options: Vec<*const i8> =
            vec![cpp_std_option_cstr.as_ptr(), include_option_cstr.as_ptr()];
        unsafe {
            let options_ptr = options.as_mut_ptr();
            let status =
                cubecl_hip_sys::hiprtcCompileProgram(program, options.len() as i32, options_ptr);
            if status != hiprtcResult_HIPRTC_SUCCESS {
                let mut log_size: usize = 0;
                let status =
                    cubecl_hip_sys::hiprtcGetProgramLogSize(program, &mut log_size as *mut usize);
                assert_eq!(
                    status, hiprtcResult_HIPRTC_SUCCESS,
                    "Should retrieve the compilation log size"
                );
                let mut log_buffer = vec![0i8; log_size];
                let status = cubecl_hip_sys::hiprtcGetProgramLog(program, log_buffer.as_mut_ptr());
                assert_eq!(
                    status, hiprtcResult_HIPRTC_SUCCESS,
                    "Should retrieve the compilation log contents"
                );
                let log = CStr::from_ptr(log_buffer.as_ptr());
                let mut message = "[Compilation Error] ".to_string();
                if log_size > 0 {
                    for line in log.to_string_lossy().split('\n') {
                        if !line.is_empty() {
                            message += format!("\n    {line}").as_str();
                        }
                    }
                } else {
                    message += "\n No compilation logs found!";
                }
                let numbered_source: String = jitc_kernel
                    .source
                    .lines()
                    .enumerate()
                    .map(|(index, line)| format!("{:<4} |   {}", index + 1, line))
                    .collect::<Vec<_>>()
                    .join("\n");
                panic!("{message}\n[Source]\n{numbered_source}");
            }
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should compile the program"
            );
        };
        // Get HIP compiled code from program
        let mut code_size: usize = 0;
        unsafe {
            let status = cubecl_hip_sys::hiprtcGetCodeSize(program, &mut code_size);
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should get size of compiled code"
            );
        }
        let mut code = vec![0i8; code_size];
        unsafe {
            let status = cubecl_hip_sys::hiprtcGetCode(program, code.as_mut_ptr());
            assert_eq!(
                status, hiprtcResult_HIPRTC_SUCCESS,
                "Should load compiled code"
            );
        }
        // Create the HIP module
        let mut module: cubecl_hip_sys::hipModule_t = std::ptr::null_mut();
        unsafe {
            let codeptr = code.as_ptr();
            let status = cubecl_hip_sys::hipModuleLoadData(&mut module, codeptr as *const _);
            assert_eq!(status, HIP_SUCCESS, "Should load compiled code into module");
        }
        // Retrieve the HIP module function
        let mut func: cubecl_hip_sys::hipFunction_t = std::ptr::null_mut();
        unsafe {
            let status =
                cubecl_hip_sys::hipModuleGetFunction(&mut func, module, func_name.as_ptr());
            assert_eq!(status, HIP_SUCCESS, "Should return module function");
        }

        // register module
        self.module_names.insert(
            kernel_id.clone(),
            HipCompiledKernel {
                _module: module,
                func,
                cube_dim: jitc_kernel.cube_dim,
                shared_mem_bytes: jitc_kernel.shared_mem_bytes,
            },
        );
    }

    fn execute_task(
        &mut self,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: Vec<HipResource>,
    ) {
        let mut bindings = resources
            .iter()
            .map(|memory| memory.binding)
            .collect::<Vec<_>>();

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;

        unsafe {
            let status = cubecl_hip_sys::hipModuleLaunchKernel(
                kernel.func,
                dispatch_count.0,
                dispatch_count.1,
                dispatch_count.2,
                cube_dim.x,
                cube_dim.y,
                cube_dim.z,
                kernel.shared_mem_bytes as u32,
                self.stream,
                bindings.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if status == cubecl_hip_sys::hipError_t_hipErrorOutOfMemory {
                panic!("Error: Cannot launch kernel (Out of memory)\n{}", kernel_id)
            }
            assert_eq!(status, HIP_SUCCESS, "Should launch the kernel");
        };
    }
}

impl HipServer {
    /// Create a new hip server.
    pub(crate) fn new(mut ctx: HipContext) -> Self {
        let logger = DebugLogger::default();
        if logger.profile_level().is_some() {
            ctx.timestamps.enable();
        }

        Self { ctx, logger }
    }

    fn get_context(&mut self) -> &mut HipContext {
        self.get_context_with_logger().0
    }

    fn get_context_with_logger(&mut self) -> (&mut HipContext, &mut DebugLogger) {
        unsafe {
            cubecl_hip_sys::hipCtxSetCurrent(self.ctx.context);
        };
        (&mut self.ctx, &mut self.logger)
    }
}

fn include_path() -> PathBuf {
    let error_msg = "
        ROCm HIP installation not found.
        Please ensure that ROCm is installed with HIP runtimes and development libraries and the ROCM_PATH or HIP_PATH environment variable is set correctly.
        Note: Default path is /opt/rocm which may not be correct.
    ";
    let path = hip_path().expect(error_msg);
    let result = path.join("include");
    let hip_include = result.join("hip");
    if !hip_include.exists() || !hip_include.is_dir() {
        panic!("{error_msg}");
    }
    result
}

fn hip_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("CUBECL_ROCM_PATH") {
        return Some(PathBuf::from(path));
    }
    if let Ok(path) = std::env::var("ROCM_PATH") {
        return Some(PathBuf::from(path));
    }
    if let Ok(path) = std::env::var("HIP_PATH") {
        return Some(PathBuf::from(path));
    }
    // Default path (only Linux is supported for now)
    Some(PathBuf::from("/opt/rocm"))
}
