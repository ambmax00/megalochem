/* Adapted from MPQC by the Valeev group
*/

#ifndef POOL_H
#define POOL_H

#include <vector>
#include <memory>
#include "omp.h"

namespace util {
	
template <typename Item>
class Pool {
private:

	std::vector<Item> m_items;
	
public:

	Pool() = delete;
	Pool(Pool const&) = delete;
	Pool &operator=(Pool const&) = delete;
	
	Pool &operator=(Pool &&) = default;
    Pool(Pool &&a) = default;
    
    explicit Pool(Item e) : 
		m_items(omp_get_max_threads(), e) {}
	
	Item &local() { return m_items[omp_get_thread_num()]; }
	
};

template <typename Item>
std::shared_ptr<Pool<Item>> make_pool(Item e) {
    return std::make_shared<Pool<Item>>(e);
}

template <typename E>
using ShrPool = std::shared_ptr<Pool<E>>;

} // end namespace

#endif  // POOL_H_
