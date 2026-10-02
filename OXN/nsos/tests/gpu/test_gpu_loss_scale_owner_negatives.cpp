
#include "gpu_parity_common.h"
#include "gpu_sparse_adam.h"
#include "gpu_execution.h"
#include <string>
using namespace nsos;
void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
#ifdef USE_CUDA
void reject_external_change(int kind){
    gpu::ExecutionContext context;gpu::ExecutionContext::Scope lane(context);
    Parameter p(Tensor::ones({2,2},Device::GPU),"negative.weight");
    p.add_grad(Tensor::ones({2,2},Device::GPU));
    GpuSparseAdam owner;std::vector<GpuSparseAdamSlot> slots{{&p,.002f,true,true}};
    owner.configure(slots);owner.abort();auto original_version=p.version;
    auto before=owner.snapshot();require(before.size()==1&&!before[0].initialized&&before[0].version==original_version,"empty owner baseline");
    if(kind==0)p.mark_updated();
    else if(kind==1)p.data=p.data.clone();
    else p.copy_data_from(Tensor::zeros({2,2},Device::GPU));
    auto external_values=p.data.cpu().clone();auto external_version=p.version;auto* external_storage=p.data.raw_data();
    bool rejected=false;
    try{owner.configure(slots);}catch(const std::logic_error& e){
        rejected=std::string(e.what())=="Sparse Adam registry weight storage/version changed";
    }
    require(rejected,"owner did not reject external weight/version/storage mutation");
    require(p.version==external_version&&p.data.raw_data()==external_storage,"rejected configure mutated externally supplied identity");
    gpu_parity_test::assert_close(p.data,external_values,0,"rejected configure changed supplied values",0);
    bool poisoned=false;
    try{(void)owner.snapshot();}catch(const std::logic_error& e){poisoned=std::string(e.what())=="Device sparse Adam recovery failed; rebuild runtime from checkpoint";}
    require(poisoned,"rejected configure owner did not remain fail-closed");
    // Do not step or reuse this invalid model/owner. External writes themselves
    // are deliberately NOT rolled back by configure; a new run is required.
    std::cout<<"owner external mutation kind="<<kind<<" rejected PASS\n";
}
#endif
int main(){return gpu_parity_test::run_parity("loss_scale_owner_negatives",[]{
#ifdef USE_CUDA
    for(int k:{0,1,2})reject_external_change(k);
#else
    throw std::runtime_error("owner negatives require GPU backend");
#endif
});}
