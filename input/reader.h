#ifndef INPUT_READER_H
#define INPUT_READER_H

#include <string>
#include <stdexcept>

#include "desc/options.h"
#include "desc/molecule.h"

class reader {
private:

	desc::molecule m_mol;
	desc::options m_opt;

public:	
	
	reader(std::string filename);
	
	desc::molecule get_mol() {
		return m_mol;
	};
	
	desc::options get_opt() {
		return m_opt;
	};
	
};

#endif
