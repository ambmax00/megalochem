#include <iostream>

constexpr int n = 5;

int main() {

	int i = 0;
	
	if constexpr (!(n == 4 && i == 1)) {
		std::cout << "P1" << std::endl;
	} else {
		std::cout << "P2" << std::endl;
	}

}
