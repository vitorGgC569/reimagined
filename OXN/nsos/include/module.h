#ifndef MODULE_H
#define MODULE_H

#include "tensor.h"
#include <map>
#include <string>
#include <vector>

namespace nsos {

// Universal Blueprint for all Layers
class Module {
public:
  virtual ~Module() = default;

  // The Contract
  virtual Tensor forward(const Tensor &input) = 0;
  virtual void to(Device device) = 0;

  // Parameter Access
  virtual std::vector<Parameter *> parameters() { return {}; }

  // State Dict export (simplified)
  virtual std::map<std::string, Tensor> state_dict() {
    std::map<std::string, Tensor> dict;
    for (auto *p : parameters()) {
      if (!p->name.empty())
        dict[p->name] = p->data;
    }
    return dict;
  }
};

} // namespace nsos

#endif
