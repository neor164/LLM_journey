//! This module contains utility structures for managing k-best elements using a binary heap.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use ordered_float::OrderedFloat; // For using f64 in BinaryHeap

/// Represents an element in the KBestNeighbors heap, pairing a distance with data.
#[derive(Debug)]
pub struct HeapElement<P> {
    pub distance: OrderedFloat<f64>, // Max-heap stores by distance
    pub data: P,
}

impl<P> PartialEq for HeapElement<P> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl<P> Eq for HeapElement<P> {}

impl<P> PartialOrd for HeapElement<P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P> Ord for HeapElement<P> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Standard BinaryHeap is a max-heap.
        // We want to keep the K smallest distances.
        // So, the heap should store elements such that the largest distance is at the top.
        self.distance.cmp(&other.distance)
    }
}

/// Manages a collection of the K "best" (e.g., smallest distance) items seen so far.
#[derive(Debug)]
pub struct KBestNeighbors<P> {
    capacity: usize,
    heap: BinaryHeap<HeapElement<P>>,
}

impl<P: Clone> KBestNeighbors<P> {
    pub fn new(capacity: usize) -> Self {
        KBestNeighbors {
            capacity,
            heap: BinaryHeap::with_capacity(capacity + 1), // +1 for easier logic
        }
    }

    pub fn add(&mut self, distance: f64, point_data: P) {
        if self.capacity == 0 { return; }
        let item = HeapElement { distance: OrderedFloat(distance), data: point_data };
        if self.heap.len() < self.capacity {
            self.heap.push(item);
        } else if item.distance < self.heap.peek().unwrap().distance { // unwrap is safe due to len check
            self.heap.pop();
            self.heap.push(item);
        }
    }

    pub fn current_farthest_distance(&self) -> Option<f64> {
        if self.heap.len() == self.capacity {
            self.heap.peek().map(|heap_elem| heap_elem.distance.0)
        } else {
            None // Not full yet, effectively infinite radius for pruning
        }
    }

    pub fn into_sorted_points(self) -> Vec<P> {
        self.heap.into_sorted_vec().into_iter().map(|elem| elem.data).collect()
    }
        /// Returns the current number of neighbors stored.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

}