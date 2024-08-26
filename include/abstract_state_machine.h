#pragma once

#include <memory>

// efsm means "edm finite state machine"
namespace efsm
{

    enum class run_state
    {
        stopped,
        starting,
        running,
        pausing,
        paused,
        resuming,
        stopping
    };

    class start_param_base
    {
    public:
        using ptr = std::shared_ptr<start_param_base>;
        start_param_base() {}
        virtual ~start_param_base() {}
    };

    class abstract_state_machine
    {
    public:
        using ptr = std::shared_ptr<abstract_state_machine>;
        abstract_state_machine() {}
        virtual ~abstract_state_machine() {}

        abstract_state_machine(const abstract_state_machine &) = delete;
        abstract_state_machine(abstract_state_machine &&) = delete;
        abstract_state_machine &operator=(const abstract_state_machine &) = delete;
        abstract_state_machine &operator=(abstract_state_machine &&) = delete;

    public:
        virtual bool restart(start_param_base::ptr) = 0;
        virtual bool pause() = 0;
        virtual bool resume() = 0;
        virtual bool stop() = 0;

        virtual bool is_stopped() const { return run_state_ == run_state::stopped; }
        virtual bool is_starting() const { return run_state_ == run_state::starting; }
        virtual bool is_running() const { return run_state_ == run_state::running; }
        virtual bool is_pausing() const { return run_state_ == run_state::pausing; }
        virtual bool is_paused() const { return run_state_ == run_state::paused; }
        virtual bool is_resuming() const { return run_state_ == run_state::resuming; }
        virtual bool is_stopping() const { return run_state_ == run_state::stopping; }

        virtual void run_once() = 0;

        virtual run_state state() const { return run_state_; }

        // if stopped, read error() to see if is stopped because of some error
        virtual int error() const { return error_; }

    protected:
        static const char *_run_state_to_str(run_state state);

    protected:
        run_state run_state_{run_state::stopped};
        int error_{0};
    };

}