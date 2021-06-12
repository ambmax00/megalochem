#ifndef MEGALOCHEM_DRIVER_HPP
#define MEGALOCHEM_DRIVER_HPP

#include <any>
#include <deque>
#include <map>
#include <string>
#include "io/data_handler.hpp"
#include "megalochem.hpp"
#include "utils/json.hpp"
#include "utils/ppdirs.hpp"

namespace megalochem {

ENUM_STRINGS(megatype, (globals, atoms, molecule, basis, hfwfn, mpwfn, adcwfn, moprint))

struct megajob {
  megatype mtype;
  nlohmann::json jdata;
};

class driver {
 private:
  world m_world;
  filio::data_io m_fh;

  util::mpi_log LOG;

  std::map<std::string, std::any> m_stack;  // variables

  std::deque<megajob> m_jobs;  // job queue

  void run_hfmod(megajob& j);

  void run_mpmod(megajob& j);

  void run_adcmod(megajob& j);

  void run_moprintmod(megajob& j);

 public:
  driver(world w, filio::data_io fh) : m_world(w), m_fh(fh), LOG(w.comm(), 0)
  {
  }

  void parse_file(std::string filename);

  void parse_json(nlohmann::json& data);

  void parse_json_section(nlohmann::json& data);

  void parse_globals(nlohmann::json& jdata);

  void parse_basis(nlohmann::json& jdata);

  void parse_atoms(nlohmann::json& jdata);

  void parse_molecule(nlohmann::json& jdata);

  void parse_hfwfn(nlohmann::json& jdata);

  void parse_mpwfn(nlohmann::json& jdata);

  void parse_adcwfn(nlohmann::json& jdata);
  
  void parse_moprint(nlohmann::json& jdata);

  void run();

  template <typename T>
  T& get(std::string key)
  {
    if (m_stack.find(key) == m_stack.end()) {
      throw std::runtime_error("Could not find " + key + " in stack!");
    }

    return std::any_cast<T&>(m_stack[key]);
  }

  ~driver()
  {
  }
};

}  // namespace megalochem

#endif
