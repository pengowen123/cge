//! A stack for use in network evaluation.

use std::ops;

#[derive(Clone, Debug, PartialEq)]
pub struct Stack(Vec<f64>);

impl ops::Deref for Stack {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Stack {
    pub fn new() -> Stack {
        Stack { 0: Vec::new() }
    }

    /// Pops `count` items off the stack and returns their sum. Returns `None` if there are fewer
    /// than `count` items on the stack.
    pub fn pop_sum(&mut self, count: usize) -> Option<f64> {
        let len = self.0.len();

        if count > len {
            return None;
        }

        Some(self.0.drain(len - count..len).sum())
    }

    pub fn pop(&mut self) -> Option<f64> {
        self.0.pop()
    }

    pub fn push(&mut self, value: f64) {
        self.0.push(value);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack() {
        let mut stack = Stack::new();

        stack.push(1.0);
        stack.push(2.0);
        stack.push(3.0);
        stack.push(4.0);

        assert_eq!(&[1.0, 2.0, 3.0, 4.0], stack.as_slice());
        assert_eq!(Some(7.0), stack.pop_sum(2));
        assert_eq!(None, stack.pop_sum(3));
        assert_eq!(Some(3.0), stack.pop_sum(2));
        assert!(stack.is_empty());
    }
}
