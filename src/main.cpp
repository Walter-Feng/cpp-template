#include <iostream>
#include <fstream>

#include <args.hxx>
#include <nlohmann/json.hpp>

#include <boost/mpi.hpp>
#include <nccl.h>

#include "version.h"
#include "util/util.h"


int main(int argc, char * argv[]) {

  boost::mpi::environment environment(argc, argv);
  boost::mpi::communicator communicator;

  std::vector<std::string> host_names;
  const auto local_host_name = environment.processor_name();
  boost::mpi::all_gather(communicator, local_host_name, host_names);

  const size_t host_index = std::find(host_names.begin(), host_names.end(),
                                      local_host_name) - host_names.begin();

  assert(host_index < host_names.size());
  const auto local_communicator = communicator.split(host_index);

  int n_devices;
  CUDACHECK(cudaGetDeviceCount(&n_devices));

  if (local_communicator.size() > n_devices) {
    throw cpp_template::Error(
        "the size of local threads exceeds allocable GPU devices");
  }

  ncclUniqueId unique_id;
  if (environment.is_main_thread()) {
    ncclGetUniqueId(&unique_id);
  }

  boost::mpi::broadcast(communicator, unique_id.internal,
                        environment.host_rank().get_value_or(0));

  CUDACHECK(cudaSetDevice(local_communicator.rank()));

  ncclComm_t nccl_communicator;
  NCCLCHECK(
      ncclCommInitRank(&nccl_communicator, local_communicator.size(), unique_id,
                       local_communicator.rank()));

  args::ArgumentParser parser(
      "This is the executable of a C++ template program "
      "Written by Rui Li (github.com/Walter-Feng)");

  args::HelpFlag help(parser, "help",
                      "Display this help menu", {'h', "help"});

  args::Positional<std::string> input_flag(parser, "input",
                                           "The input file (in json format)");

  args::ValueFlag<std::string> str_input(parser, "string",
                                         "String form of json input", {'s'});

  args::Flag version_flag(parser, "version", "Check the version", {'v'});

  /////////////// Error Handling ///////////////
  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Help &) {
    std::cout << parser << std::endl;
    return 0;
  } catch (const args::ParseError & error) {
    std::cout << error.what() << std::endl;
    std::cout << parser << std::endl;
    return 1;
  } catch (const args::ValidationError & error) {
    std::cout << error.what() << std::endl;
    std::cout << parser << std::endl;
    return 2;
  }

  /////////////// Print version ///////////////

  if (args::get(version_flag)) {
    std::cout << "version: " << VERSION << std::endl;
    return 0;
  }

  nlohmann::json input;
  if (str_input) {
    input = nlohmann::json::parse(args::get(str_input));
  } else {
    const std::string input_filename = args::get(input_flag);
    if (input_filename.empty()) {
      // If nothing is input, the help text is called
      std::cout << parser << std::endl;
      return 0;
    }
    std::ifstream input_file_stream(args::get(input_flag));

    input_file_stream >> input;
  }
}