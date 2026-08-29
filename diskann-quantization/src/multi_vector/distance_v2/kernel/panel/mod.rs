/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) mod maxsim;

pub(crate) trait Kernel {
    fn run(&mut self);
}
