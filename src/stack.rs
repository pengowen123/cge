//! A stack for use in network evaluation.

use num_traits::Float;

use std::ops;

#[derive(Clone, Debug, PartialEq)]
pub struct Stack<T>(Vec<T>);

impl<T> ops::Deref for Stack<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Stack<T> {
    pub fn new() -> Self {
        Stack(Vec::new())
    }

    /// Pops `count` items off the stack and returns their sum. Returns `None` if there are fewer
    /// than `count` items on the stack.
    pub fn pop_sum(&mut self, count: usize) -> Option<T>
    where
        T: Float,
    {
        let len = self.0.len();

        if count > len {
            return None;
        }

        Some(
            self.0
                .drain(len - count..len)
                .fold(T::zero(), |acc, x| acc + x),
        )
    }

    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    pub fn push(&mut self, value: T) {
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
