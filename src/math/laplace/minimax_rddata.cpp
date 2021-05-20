#include "math/laplace/minimax.hpp"
#include <fstream>
#include <string>
#include <stdexcept>

#cmakedefine LAPLACE_ROOT "@LAPLACE_ROOT@/init_para.txt"

namespace megalochem {
	
namespace math {

std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

std::tuple<int,double> evaluate_xkstr(std::string xkstr) {
	
	auto tokens = split(xkstr, '_');
	
	auto nlap_str = tokens[1].substr(2,2);
	auto R0_str = tokens[2];
	
	int nlap = std::stoi(nlap_str);
	double R0 = 0.0;
	
	if (R0_str.find("E") != std::string::npos) {
		auto be_str = split(R0_str, 'E');
		int base = std::stoi(be_str[0]);
		int expo = std::stoi(be_str[1]);
		R0 = base * std::pow(10, (double)expo);
	} else {
		R0 = std::stod(R0_str) / 1000.0;
	}
	
	return std::make_tuple(nlap, R0);
	
}	

std::tuple<Eigen::VectorXq,Eigen::VectorXq> 
	minimax::read_guess(double R, int k)
{
	
	os<1>("Reading initial guess for omegas/alphas\n");
	
	std::string filename = LAPLACE_ROOT;
	
	std::ifstream infile(filename);
	
	if (!infile.is_open()) {
		throw std::runtime_error("MINIMAX: Could not open file");
	}
	
	std::string line;
	
	Eigen::VectorXq omega_guess(k);
	Eigen::VectorXq alpha_guess(k);
	
	// skip first four lines
	for (int ii = 0; ii != 4; ++ii) {
		std::getline(infile, line);
	}
	
	double R0_chosen = -1.0;
	int k_recommended = 0;
		
	while (line != "$end") {
		// read header
		std::getline(infile, line);
				
		std::string xkstr = line.substr(0,line.size());
		
		os<1>(xkstr, '\n');
		
		auto [nlap, R0] = evaluate_xkstr(xkstr);
		
		os<1>("nlap/R0: ", nlap, " ", R0, '\n');
		
		if (nlap == k && R0 >= R) break;
		
		if (nlap != k && R0 <= R) k_recommended = nlap-1;
		
		if (nlap == k) R0_chosen = R0;
		
		bool read = (nlap == k);
						
		for (int ii = 0; ii != nlap; ++ii) {
			std::getline(infile, line);
			auto linesplit = split(line, ' ');
			if (read) {
				int pos = (linesplit[0] == "") ? 1 : 0;
				convert_from_string(omega_guess(ii), linesplit[pos].c_str()); 
			}
		}
		
		for (int ii = 0; ii != nlap; ++ii) {
			std::getline(infile, line);
			auto linesplit = split(line, ' ');
			if (read) {
				int pos = (linesplit[0] == "") ? 1 : 0;
				convert_from_string(alpha_guess(ii), linesplit[pos].c_str()); 
			}
		}
				
	}
	
	if (R0_chosen < 0.0 && k_recommended == 0) {
		throw std::runtime_error("R value is too small!");
	} 
	
	if (R0_chosen < 0.0 && k_recommended != 0) {
		throw std::runtime_error("R value too small. "
			"Use " + std::to_string(k_recommended) + " quadrature points instead.");
	}
	
	os<1>("Reading R0 = ", R0_chosen, '\n');
	
	os<1>("Omega/Alpha:\n");
	os<1>(omega_guess, '\n');
	os<1>(alpha_guess, '\n');
	
	return std::make_tuple(omega_guess, alpha_guess);
	
}

} // namespace math 

} // namespace megalochem
