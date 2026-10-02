#include "gpu_parity_common.h"
#include "gpu_execution.h"
#include "gpu_gemm_provider.h"
#include "nsos/determinism.h"
#include "jamba.h"
#include "trainer.h"
#include <atomic>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

using namespace nsos;
namespace {
void require(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
template<class F> void rejected(F f,const char* why) {bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed,why);}
void fill(Tensor& t,float scale=.13f) {for(std::size_t i=0;i<t.size;++i)t.data()[i]=scale*std::sin(float(i+1)*.173f);}
std::vector<float> values(const Tensor& t) {auto h=t.cpu();return {h.data(),h.data()+h.size};}
void close(const Tensor& t,const std::vector<float>& expected,const char* why,float tol=3e-5f) {
    auto got=values(t);require(got.size()==expected.size(),"BLAS oracle extent");
    for(std::size_t i=0;i<got.size();++i)if(!std::isfinite(got[i])||std::abs(got[i]-expected[i])>tol*(1+std::abs(expected[i])))
        throw std::runtime_error(std::string(why)+" index="+std::to_string(i));
}
std::vector<float> oracle(const Tensor& a,const Tensor& b,bool ta,bool tb,int M,int N,int K) {
    std::vector<float> out(std::size_t(M)*N);
    for(int m=0;m<M;++m)for(int n=0;n<N;++n){double sum=0;for(int k=0;k<K;++k)
        sum+=double(a.data()[ta?std::size_t(k)*M+m:std::size_t(m)*K+k])*
            b.data()[tb?std::size_t(n)*K+k:std::size_t(k)*N+n];out[std::size_t(m)*N+n]=float(sum);}
    return out;
}
void* current_handle(gpu::ExecutionContext& context) {
    gpu::ExecutionContext::ClassicBlasLease lease(context);
    require(lease.handle()!=nullptr,"classic BLAS unavailable");
    cudaStream_t stream=nullptr;
#if defined(NSOS_GPU_BACKEND_HIP)
    require(hipblasGetStream(reinterpret_cast<cublasHandle_t>(lease.handle()),&stream)==CUBLAS_STATUS_SUCCESS,"query HIP BLAS stream");
#else
    require(cublasGetStream(reinterpret_cast<cublasHandle_t>(lease.handle()),&stream)==CUBLAS_STATUS_SUCCESS,"query CUDA BLAS stream");
#endif
    require(stream==gpu::current_stream(),"handle migrated from its owner stream");
    return lease.handle();
}
Tensor product(const Tensor& a,const Tensor& b,bool ta,bool tb) {
    return ta?matmul_tn(a,b):tb?matmul_nt(a,b):a.matmul(b);
}
void alternating_contexts() {
    auto& fallback=gpu::current_execution_context();void* default_handle=current_handle(fallback);
    for(int cycle=0;cycle<64;++cycle) {
        auto a=std::make_unique<gpu::ExecutionContext>(),b=std::make_unique<gpu::ExecutionContext>();
        const int M=17+cycle%3*16,N=33+cycle%5*8,K=35+cycle%4*16;
        void* handle_a=nullptr;void* handle_b=nullptr;cudaStream_t stream_a=nullptr,stream_b=nullptr;
        for(const auto flags:{std::pair{false,false},std::pair{false,true},std::pair{true,false}}) {
            Tensor lhs(flags.first?std::vector<int>{K,M}:std::vector<int>{M,K});fill(lhs);
            Tensor rhs(flags.second?std::vector<int>{N,K}:std::vector<int>{K,N});fill(rhs,.17f);
            auto ga=lhs.to(Device::GPU),gb=rhs.to(Device::GPU);Tensor ya,yb,repeated;
            {
                gpu::ExecutionContext::Scope scope(*a);stream_a=gpu::current_stream();
                const auto handle=current_handle(*a);if(handle_a)require(handle==handle_a,"owner A handle changed");else handle_a=handle;
                ya=product(ga,gb,flags.first,flags.second);
                {
                    // Nested lane changes must leave the two handles bound to
                    // their original streams, and preserve producer ordering.
                    gpu::ExecutionContext::Scope nested(*b);stream_b=gpu::current_stream();
                    const auto handle=current_handle(*b);if(handle_b)require(handle==handle_b,"owner B handle changed");else handle_b=handle;
                    yb=product(ga,gb,flags.first,flags.second);
                }
                require(current_handle(*a)==handle_a,"nested scope reinterpreted A handle");
                repeated=product(ga,gb,flags.first,flags.second);
            }
            require(handle_a!=handle_b&&handle_a!=default_handle&&handle_b!=default_handle,"contexts share mutable BLAS allocator");
            require(stream_a!=stream_b,"private contexts share a stream");
            close(ya,oracle(lhs,rhs,flags.first,flags.second,M,N,K),"owner A GEMM");
            const auto av=values(ya),bv=values(yb),rv=values(repeated);
            require(av.size()==bv.size()&&av.size()==rv.size()&&
                std::memcmp(av.data(),bv.data(),av.size()*sizeof(float))==0&&
                std::memcmp(av.data(),rv.data(),av.size()*sizeof(float))==0,"context alternation changed deterministic GEMM");
        }
        // Destroy A first on alternating rounds, B first otherwise, then force
        // a standalone GEMM and allocate a fresh private stream/BLAS allocator.
        if(cycle%2){a.reset();b.reset();}else{b.reset();a.reset();}
        require(current_handle(fallback)==default_handle,"destroyed model invalidated standalone handle");
        Tensor x({17,35}),w({35,33});fill(x);fill(w,.17f);
        close(x.to(Device::GPU).matmul(w.to(Device::GPU)),oracle(x,w,false,false,17,33,35),"standalone after destruction");
    }
}
void freeze_and_graph() {
    gpu::ExecutionContext cold;
    {gpu::ExecutionContext::Scope scope(cold);cold.freeze(true);
        rejected([&]{current_handle(cold);},"frozen context lazily allocated BLAS handle");
        cold.freeze(false);require(current_handle(cold)!=nullptr,"cold freeze rejection poisoned recovery");}
    gpu::ExecutionContext context;
    Tensor a({17,35}),b({35,33});fill(a);fill(b,.17f);
    auto da=a.to(Device::GPU),db=b.to(Device::GPU);auto output=Tensor::zeros({17,33},Device::GPU);
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    gpu::ExecutionContext::Scope scope(context);
    const float alpha=1,beta=0;
    const auto enqueue=[&] {
        gpu::ExecutionContext::ClassicBlasLease lease(context);
        const auto handle=reinterpret_cast<cublasHandle_t>(lease.handle());
        require(handle!=nullptr,"graph BLAS unavailable");
        require(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,33,17,35,&alpha,db.raw_data(),33,
            da.raw_data(),35,&beta,output.raw_data(),33)==CUBLAS_STATUS_SUCCESS,"graph BLAS GEMM enqueue");
    };
    enqueue();gpu_parity_test::cuda_sync_or_throw("warm BLAS graph");context.freeze(true);
    require(cudaStreamBeginCapture(gpu::current_stream(),cudaStreamCaptureModeRelaxed)==cudaSuccess,"begin BLAS graph");
    enqueue();
    require(cudaStreamEndCapture(gpu::current_stream(),&graph)==cudaSuccess&&graph!=nullptr,"end BLAS graph");
#if defined(NSOS_GPU_BACKEND_HIP) || CUDART_VERSION >= 12000
    require(cudaGraphInstantiate(&executable,graph,0)==cudaSuccess,"instantiate BLAS graph");
#else
    require(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0)==cudaSuccess,"instantiate BLAS graph");
#endif
    for(int i=0;i<3;++i)require(cudaGraphLaunch(executable,gpu::current_stream())==cudaSuccess,"launch BLAS graph");
    gpu_parity_test::cuda_sync_or_throw("replay BLAS graph");
    close(output,oracle(a,b,false,false,17,33,35),"BLAS graph parity");
    require(cudaGraphExecDestroy(executable)==cudaSuccess,"destroy BLAS graph executable");
    require(cudaGraphDestroy(graph)==cudaSuccess,"destroy BLAS graph");context.freeze(false);
}
void shared_lane_threads() {
    gpu::ExecutionContext shared;std::atomic<int> completed{0};std::atomic<bool> failed{false};
    const auto worker=[&] {
        try {
            gpu_parity_test::require_cuda_device("blas_context_worker");
            Tensor x({17,35}),w({35,33});fill(x);fill(w,.17f);const auto expected=oracle(x,w,false,false,17,33,35);
            auto dx=x.to(Device::GPU),dw=w.to(Device::GPU);
            for(int i=0;i<8;++i){gpu::ExecutionContext::Scope scope(shared);
                close(dx.matmul(dw),expected,"serialized shared lane");current_handle(shared);}
            ++completed;
        }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;failed=true;}
    };
    std::thread first(worker),second(worker);first.join();second.join();
    require(!failed&&completed==2,"shared context was not serialized across worker threads");
}
void checkpoint_recreation() {
    ModelConfig config;config.num_layers=1;config.d_model=32;config.vocab_size=32;config.n_heads=4;config.n_kv_heads=2;
    config.architecture_schema_version=2;config.mamba2_faithful=true;config.mamba_expand=1;
    config.mamba_state_expansion=true;config.mamba_d_state=8;config.mamba_head_dim=32;config.mamba_n_groups=1;
    config.use_moe=false;config.use_ttt=false;config.attention_period=8;config.max_context_tokens=16;
    const auto root=std::filesystem::temp_directory_path();
    for(int i=0;i<4;++i) {
        determinism::DeterminismManager::instance().set_global_seed(20261001+i);
        JambaModel original(config,Device::GPU);Trainer trainer(&original,.002f);
        trainer.phase_scheduler.progressive_qat_enabled=false;trainer.weight_decay=.01f;
        require(std::isfinite(trainer.train_step({1,2,3,4},{2,3,4,5})),"lifecycle training invalid");
        const auto model_path=(root/("nsos-blas-context-"+std::to_string(i)+".model")).string();
        const auto state_path=(root/("nsos-blas-context-"+std::to_string(i)+".trainer")).string();
        auto snapshot=trainer.capture_checkpoint_snapshot();snapshot->write(model_path,state_path);
        JambaModel restored(config,Device::GPU);restored.load(model_path);Trainer resumed(&restored,.04f);
        resumed.load_training_state(state_path,model_path);
        const float expected=trainer.train_step({1,2,3,4},{2,3,4,5}),actual=resumed.train_step({1,2,3,4},{2,3,4,5});
        require(std::isfinite(actual)&&std::abs(expected-actual)<1e-6f,"lifecycle checkpoint loss diverged");
        require(trainer.global_step_count==resumed.global_step_count&&trainer.tokens_committed==resumed.tokens_committed,"lifecycle checkpoint progress diverged");
        auto a=original.parameters(),b=restored.parameters();require(a.size()==b.size(),"lifecycle checkpoint registry");
        for(std::size_t j=0;j<a.size();++j)close(b[j]->data,values(a[j]->data),"lifecycle checkpoint parameters",2e-6f);
        std::filesystem::remove(model_path);std::filesystem::remove(state_path);
    }
}
}
int main() {
    return gpu_parity_test::run_parity("blas_context_lifetime",[] {
        set_matmul_precision_mode(0);determinism::set_deterministic_reductions(true);
        gpu::require_classic_gemm_provider();
        alternating_contexts();freeze_and_graph();shared_lane_threads();checkpoint_recreation();
        gpu_parity_test::cuda_sync_or_throw("BLAS context lifetime final drain");
        std::cout<<"[BLAS lifetime] distinct_fixed_stream_handles NN/NT/TN oracle deterministic reuse destroy/recreate graph freeze threads checkpoint PASS"<<std::endl;
    });
}
