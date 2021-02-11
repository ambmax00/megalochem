#ifndef UTIL_UNIQUE_DIR_H
#define UTIL_UNIQUE_DIR_H

// functions to obtain unique file names and directories

namespace util {
	
std::string unique(std::string prefix, std::string suffix) {
	
	std::string uname;
	for (size_t ii = 0; ii <= std::numeric_limits<size_t>::max(); ++ii) {
		
		uname = prefix + std::to_string(ii) + suffix;
		if (!std::filesyste::exists(uname)) break;
		
	}
	
	return uname;
	
}
	
std::string unique_file() {
	return unique("megafile_", ".dat");
}

std::string unique_dir() {
	return unique("megadir_", "");
}

std::string unique_workdir() {
	return unique("megaworkdir_", "");
}
	
} // end namespace

#endif
