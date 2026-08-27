// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_RUNTIME_WORKSPACE_HPP
#define SCIVOL_EXPERIMENT_RUNTIME_WORKSPACE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace vlk::experiment::runtime {

class WorkspaceSizer {
public:
    template<class T>
    bool add(std::size_t count) noexcept
    {
        const std::size_t aligned = align_up(size_, alignof(T));
        if (aligned == invalid || count >
                (std::numeric_limits<std::size_t>::max() - aligned) / sizeof(T)) {
            valid_ = false;
            return false;
        }
        size_ = aligned + count * sizeof(T);
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return valid_ ? size_ : 0u;
    }

private:
    static constexpr std::size_t invalid =
        std::numeric_limits<std::size_t>::max();

    static std::size_t align_up(
        std::size_t value, std::size_t alignment) noexcept
    {
        const std::size_t remainder = value % alignment;
        if (remainder == 0u)
            return value;
        const std::size_t increment = alignment - remainder;
        return value <= std::numeric_limits<std::size_t>::max() - increment
            ? value + increment
            : invalid;
    }

    std::size_t size_ = 0u;
    bool valid_ = true;
};

class WorkspaceArena {
public:
    WorkspaceArena(void* memory, std::size_t bytes) noexcept
        : cursor_(static_cast<std::byte*>(memory)), remaining_(bytes)
    {
    }

    template<class T>
    T* take(std::size_t count) noexcept
    {
        if (cursor_ == nullptr)
            return nullptr;
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(cursor_);
        const std::size_t padding =
            (alignof(T) - address % alignof(T)) % alignof(T);
        if (padding > remaining_
            || count > (remaining_ - padding) / sizeof(T)) {
            return nullptr;
        }
        cursor_ += padding;
        remaining_ -= padding;
        T* result = reinterpret_cast<T*>(cursor_);
        const std::size_t consumed = count * sizeof(T);
        cursor_ += consumed;
        remaining_ -= consumed;
        return result;
    }

private:
    std::byte* cursor_ = nullptr;
    std::size_t remaining_ = 0u;
};

[[nodiscard]] inline bool checked_product(
    std::size_t left,
    std::size_t right,
    std::size_t& result) noexcept
{
    if (left != 0u
        && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

[[nodiscard]] inline bool checked_sum(
    std::size_t left,
    std::size_t right,
    std::size_t& result) noexcept
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

}  // namespace vlk::experiment::runtime

#endif
