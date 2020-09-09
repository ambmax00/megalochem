#ifndef UTILS_PPDIRS_H
#define UTILS_PPDIRS_H

#define make_param(structname,name,type,reqopt,refval) \
	private: \
		reqopt < type, refval > c_##name; \
	public: \
		inline structname & name (reqopt < type, refval > i_##name) { \
			c_##name = i_##name; \
			return *this; \
		}

#endif
