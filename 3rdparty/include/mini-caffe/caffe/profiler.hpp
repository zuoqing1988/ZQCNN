#ifndef CAFFE_PROFILER_HPP_
#define CAFFE_PROFILER_HPP_

#include <vector>
#include <string>

#include "caffe/base.hpp"

namespace caffe {

/*!
 * \brief Profiler for Caffe, don't enable Profiler in Multi-thread Env
 *  This class is used to profile a range of source code as a scope.
 *  The basic usage is like below.
 *
 * ```
 * Profiler *profiler = Profiler::Get();
 * profiler->ScopeStart("scope1");
 * ...
 * ...
 * profiler->ScopeEnd();
 * ```
 *
 * Scope represents a range of source code. Nested scope is also supported.
 * Dump profile into a json file, then we can view the data from google chrome
 * in chrome://tracing/
 */
class CAFFE_API Profiler {
public:
  /*! \brief get global instance */
  static Profiler *Get();
  /*!
   * \brief start a scope
   * \param name scope name
   */
  void ScopeStart(const char *name);
  /*!
   * \brief end a scope
   */
  void ScopeEnd();
  /*!
   * \brief dump profile data
   * \param fn file name
   */
  void DumpProfile(const char *fn) const;
  /*! \brief turn on profiler */
  void TurnON() {
    CHECK_EQ(state_, kNotRunning) << "Profile is already running.";
    state_ = kRunning;
  }
  /*! \brief turn off profiler */
  void TurnOFF() {
    CHECK_EQ(state_, kRunning) << "Profile is not running.";
    CHECK(scope_stack_.empty()) << "Profile scope stack is not empty, with size = "
        << scope_stack_.size();
    state_ = kNotRunning;
  }
  /*! \brief timestamp, return in microseconds */
  uint64_t Now() const;

private:
  Profiler();
  DISABLE_COPY_AND_ASSIGN(Profiler);

private:
  enum State {
    kRunning,
    kNotRunning,
  };
  struct Scope {
    std::string name;
    uint64_t start_microsec = 0;
    uint64_t end_microsec = 0;
  };
  typedef std::shared_ptr<Scope> ScopePtr;
  /*! \brief scope stack for nested scope */
  std::vector<ScopePtr> scope_stack_;
  /*! \brief all scopes used in profile, not including scopes in stack */
  std::vector<ScopePtr> scopes_;
  /*! \brief init timestamp */
  uint64_t init_;
  /*! \brief profile state */
  State state_;
};  // class Profiler

}  // namespace

#endif  // CAFFE_PROFILER_HPP_
