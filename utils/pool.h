/* Taken from MPQC by the Valeev group
   Slightly modified
*/

#ifndef POOL_H
#define POOL_H

#include <memory>

#include <tbb/enumerable_thread_specific.h>

namespace util {

/// A pool of thread-specific objects
template <typename Item>
class TSPool {
  public:
    /// Don't allow copies or default initialization.
    TSPool() = delete;
    TSPool(TSPool const &) = delete;
    TSPool &operator=(TSPool const &) = delete;

    TSPool &operator=(TSPool &&) = default;
    TSPool(TSPool &&a) = default;

    /// Initializes the pool with a single @c Item
    explicit TSPool(Item e)
        : item_(std::move(e)), items_(item_) {}
    /// @return reference to the thread-local @c Item instance.
    Item &local() { return items_.local(); }

  private:
    Item item_;
    tbb::enumerable_thread_specific<Item> items_;
};

template <typename Item>
std::shared_ptr<TSPool<Item>> make_pool(Item e) {
    return std::make_shared<TSPool<Item>>(std::move(e));
}

template <typename E>
using ShrPool = std::shared_ptr<TSPool<E>>;

} // end namespace

#endif  // POOL_H_
