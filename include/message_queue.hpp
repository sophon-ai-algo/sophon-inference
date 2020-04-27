#pragma once
#include <array>
#include <mutex>

/// Namespace containing all symbols from the sail library.
namespace sail {

template<typename T, std::size_t N>
class CircularQueue {
 public:
  CircularQueue();

  bool push(T const& item);
  bool push(T&& item);

  template<typename U>
  bool pop(U& item);
  bool pop();

  inline bool full() const;
  inline bool empty() const;

  inline size_t size() const;

 protected:
  std::array<T, N> queue_;
  std::size_t front_;
  std::size_t back_;
  std::size_t size_;
};

template<typename T, std::size_t N>
CircularQueue<T, N>::CircularQueue()
  : front_(0), back_(0), size_(0){
}

template<typename T, std::size_t N>
bool CircularQueue<T, N>::full() const {
  return size_ == N;
}

template<typename T, std::size_t N>
bool CircularQueue<T, N>::empty() const {
  return size_ == 0;
}

template<typename T, std::size_t N>
size_t CircularQueue<T, N>::size() const {
  return size_;
}

template<typename T, std::size_t N>
bool CircularQueue<T, N>::push(T const& item) {
  if (this->full()) {
    return false;
  }
  queue_[back_] = item;
  back_ = (back_ + 1) % N;
  ++size_;
  return true;
}

template<typename T, std::size_t N>
bool CircularQueue<T, N>::push(T&& item) {
  if (this->full()) {
    return false;
  }
  queue_[back_] = std::move(item);
  back_ = (back_ + 1) % N;
  ++size_;
  return true;
}

template<typename T, std::size_t N>
template<typename U>
bool CircularQueue<T, N>::pop(U& item) {
  if (this->empty()) {
    return false;
  }
  item = std::move(queue_[front_]);
  front_ = (front_ + 1) % N;
  --size_;
  return true;
}

template<typename T, std::size_t N>
bool CircularQueue<T, N>::pop() {
  if (this->empty()) {
    return false;
  }
  front_ = (front_ + 1) % N;
  --size_;
  return true;
}

template<typename T, std::size_t N>
class BlockingQueue : public CircularQueue<T, N> {
 public:
  BlockingQueue();

  bool push(T const& item);
  bool push(T&& item);

  template<typename U>
  bool pop(U& item);
  bool pop();

 protected:
  std::mutex mutex_;
};

template<typename T, std::size_t N>
BlockingQueue<T, N>::BlockingQueue()
  : CircularQueue<T, N>::CircularQueue() {
}

template<typename T, std::size_t N>
bool BlockingQueue<T, N>::push(T const& item) {
  std::lock_guard<std::mutex> guard(mutex_);
  return CircularQueue<T, N>::push(item);
}

template<typename T, std::size_t N>
bool BlockingQueue<T, N>::push(T&& item) {
  std::lock_guard<std::mutex> guard(mutex_);
  return CircularQueue<T, N>::push(std::move(item));
}

template<typename T, std::size_t N>
template<typename U>
bool BlockingQueue<T, N>::pop(U& item) {
  std::lock_guard<std::mutex> guard(mutex_);
  return CircularQueue<T, N>::pop(item);
}

template<typename T, std::size_t N>
bool BlockingQueue<T, N>::pop() {
  std::lock_guard<std::mutex> guard(mutex_);
  return CircularQueue<T, N>::pop();
}

enum class DropStrategy {
  DROP_FRONT,
  DROP_BACK
};

template<typename T, std::size_t N>
class MessageQueue : public BlockingQueue<T, N> {
 public:
  MessageQueue(DropStrategy strategy=DropStrategy::DROP_FRONT);

  bool push(T const& item);
  bool push(T&& item);

 protected:
  DropStrategy strategy_;
};

template<typename T, std::size_t N>
MessageQueue<T, N>::MessageQueue(DropStrategy strategy)
  : BlockingQueue<T, N>::BlockingQueue(), strategy_(strategy) {
}

template<typename T, std::size_t N>
bool MessageQueue<T, N>::push(T const& item) {
  while (this->full()) {
    if (strategy_ == DropStrategy::DROP_BACK) {
      return true;
    }
    if (!this->pop()) {
      return false;
    }
  }
  return BlockingQueue<T, N>::push(item);
}

template<typename T, std::size_t N>
bool MessageQueue<T, N>::push(T&& item) {
  while (this->full()) {
    if (strategy_ == DropStrategy::DROP_BACK) {
      return true;
    }
    if (!this->pop()) {
      return false;
    }
  }
  return BlockingQueue<T, N>::push(std::move(item));
}

}  // namespace sail
