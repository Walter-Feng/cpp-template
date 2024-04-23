#include <iostream>
#include <fstream>

#include <args.hxx>
#include <nlohmann/json.hpp>

#include "version.h"

int main(const int argc, const char * argv[]) {

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