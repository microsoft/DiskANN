/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

#[derive(Clone)]
pub(crate) struct Sequencer(Arc<SequencerInner>);

struct SequencerInner {
    state: Mutex<State>,
    condvar: Condvar,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Empty,
    Parked(usize),
    Released(usize),
}

impl Sequencer {
    pub(crate) fn new() -> Self {
        Self(Arc::new(SequencerInner {
            state: Mutex::new(State::Empty),
            condvar: Condvar::new(),
        }))
    }

    pub(crate) fn wait_for(&self, stage: usize) {
        let mut state = self.0.state.lock();
        if stage == 0 {
            assert_eq!(*state, State::Empty)
        } else {
            assert_eq!(*state, State::Released(stage - 1))
        }

        *state = State::Parked(stage);
        self.0.condvar.notify_all();
        self.0
            .condvar
            .wait_while(&mut state, move |s| *s != State::Released(stage));
    }

    pub(crate) fn advance_past(&self, stage: usize) {
        let mut state = self.0.state.lock();
        self.0
            .condvar
            .wait_while(&mut state, move |s| Self::check_release(*s, stage));
        *state = State::Released(stage);
        self.0.condvar.notify_all();
    }

    pub(crate) fn until_waiting_for(&self, stage: usize) {
        let mut state = self.0.state.lock();
        if *state != State::Parked(stage) {
            self.0
                .condvar
                .wait_while(&mut state, move |s| Self::check_release(*s, stage))
        }
    }

    fn check_release(current: State, stage: usize) -> bool {
        match current {
            State::Empty => {
                assert_eq!(stage, 0);
                true
            }
            State::Released(s) => {
                if s + 1 != stage {
                    panic!("observed {:?} while releasing stage {}", current, stage);
                }
                true
            }
            State::Parked(s) => {
                if s != stage {
                    panic!("observed {:?} while releasing stage {}", current, stage)
                }
                false
            }
        }
    }
}
