#![feature(test)]

extern crate test;

use bitmaps::Bitmap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem;

// TODO: Make the hop limit a parameter.
const HOP_LIMIT: usize = 32;

#[derive(Clone, Debug)]
struct Entry<K, V>
where
    K: Hash + Eq,
{
    key: K,
    value: V,
}

#[derive(Clone, Debug)]
struct Bucket<K, V>
where
    K: Hash + Eq,
{
    hop_info: Bitmap<HOP_LIMIT>,
    entry: Option<Entry<K, V>>,
}

#[derive(Clone, Debug)]
pub struct HsHashMap<K, V>
where
    K: Hash + Eq,
{
    buckets: Vec<Bucket<K, V>>,
    length: usize,
}

impl<K, V> Bucket<K, V>
where
    K: Hash + Eq,
{
    fn new() -> Bucket<K, V> {
        Bucket {
            hop_info: Bitmap::new(),
            entry: None,
        }
    }

    // TODO: This is ambiguous. Is the slot itself vacant, or is the entire
    // logical bucket vacant?
    fn is_vacant(&self) -> bool {
        self.entry.is_none()
    }

    fn is_full(&self) -> bool {
        self.hop_info.is_full()
    }
}

// TODO: Add deletion.
// TODO: Optimize.
impl<K, V> HsHashMap<K, V>
where
    K: Hash + Eq,
{
    // Public functions.
    pub fn new() -> HsHashMap<K, V> {
        let mut buckets: Vec<Bucket<K, V>> = vec![];
        buckets.resize_with(HOP_LIMIT, Bucket::new);
        HsHashMap {
            buckets: buckets,
            length: 0,
        }
    }

    // This is actually insert_or_assign() in C++.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        // First, try to find `k`.
        if let Some(val) = self.get_mut(&k) {
            return Some(mem::replace(val, v)); // std::exchange
        }
        // If `k` isn't present, find the closest empty slot, and insert it.
        let bucket_idx = self.get_bucket(&k);
        if self.buckets[bucket_idx].is_full() {
            self.rehash(self.capacity() * 2);
            return self.insert(k, v);
        }
        let capacity = self.capacity();
        let vacant_hop = (0..capacity).find(|hop| {
            let entry_idx = (bucket_idx + hop) % capacity;
            self.buckets[entry_idx].entry.is_none()
        });
        if vacant_hop.is_none() {
            self.rehash(self.capacity() * 2);
            return self.insert(k, v);
        }
        // Keep moving the empty slot closer to bucket_idx.
        let mut hop = vacant_hop.unwrap();
        while hop >= HOP_LIMIT {
            match self.move_closer(bucket_idx, hop) {
                None => {
                    self.rehash(self.capacity() * 2);
                    return self.insert(k, v);
                }
                Some(new_hop) => {
                    hop = new_hop;
                }
            }
        }
        let buckets = &mut self.buckets;
        let entry_idx = (bucket_idx + hop) % capacity;
        debug_assert!(buckets[entry_idx].is_vacant());
        debug_assert!(!buckets[bucket_idx].hop_info.get(hop));
        buckets[bucket_idx].hop_info.set(hop, true);
        buckets[entry_idx].entry = Some(Entry { key: k, value: v });
        self.length += 1;
        None
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        let bucket_idx = self.get_bucket(&k);
        let hop_info = &self.buckets[bucket_idx].hop_info;
        for hop in (0..HOP_LIMIT).filter(|hop| hop_info.get(*hop)) {
            let entry_idx = (bucket_idx + hop) % self.capacity();
            match &self.buckets[entry_idx].entry {
                None => {
                    // TODO: Can we relax this such that, when deleting an 
                    // item, we don't backtrack to reset all the bitmaps?
                    std::debug_assert!(false, "Bucket should not be empty.");
                    continue;
                }
                Some(Entry { key, value }) if key == k => {
                    return Some(&value);
                }
                _ => {}
            }
        }
        None
    }

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        // Evil hack for DRY-ness. C++ const_cast.
        unsafe { mem::transmute(self.get(k)) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.buckets.len()
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn clear(&mut self) {
        self.buckets.clear();
        self.buckets.resize_with(HOP_LIMIT, Bucket::new);
        self.length = 0;
    }

    // Private functions.
    fn get_bucket(&self, k: &K) -> usize {
        self.get_hash(&k) as usize % self.capacity()
    }

    // TODO: Make the hash function a parameter.
    fn get_hash(&self, k: &K) -> u64 {
        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        hasher.finish()
    }

    fn rehash(&mut self, new_capacity: usize) {
        // println!("Rehashing from {} to {}.", self.capacity(), new_capacity);
        let buckets = mem::replace(&mut self.buckets, vec![]);
        let original_length = self.length;
        self.buckets.resize_with(new_capacity, Bucket::new);
        self.length = 0;
        for bucket in buckets {
            if let Some(entry) = bucket.entry {
                self.insert(entry.key, entry.value);
            }
        }
        debug_assert_eq!(self.length, original_length);
    }

    // Moves the empty slot at `hop` closer to `origin_idx`.
    fn move_closer(&mut self, origin_idx: usize, current_hop: usize) -> Option<usize> {
        debug_assert!(current_hop >= HOP_LIMIT);
        let current_idx = (origin_idx + current_hop) % self.capacity();
        debug_assert!(self.buckets[current_idx].is_vacant());
        for candidate_hop in (current_hop - HOP_LIMIT + 1)..current_hop {
            let candidate_idx = (origin_idx + candidate_hop) % self.capacity();
            let candidate_to_current_delta = current_hop - candidate_hop;
            // Hop info for current_hop must be empty.
            debug_assert!(candidate_to_current_delta < HOP_LIMIT);
            debug_assert!(!self.buckets[candidate_idx]
                .hop_info
                .get(candidate_to_current_delta));
            // Find the earliest slot for `entry_idx` that isn't empty.
            if let Some(victim_hop) = self.buckets[candidate_idx].hop_info.first_index() {
                if victim_hop >= candidate_to_current_delta {
                    continue;
                }
                // Swap the victim entry with the empty slot.
                let victim_idx = (candidate_idx + victim_hop) % self.capacity();
                let victim_entry = mem::take(&mut self.buckets[victim_idx].entry);
                debug_assert!(self.buckets[victim_idx].is_vacant());
                self.buckets[current_idx].entry = victim_entry;
                self.buckets[candidate_idx].hop_info.set(victim_hop, false);
                self.buckets[candidate_idx]
                    .hop_info
                    .set(candidate_to_current_delta, true);
                // println!(
                //     "Moved an empty slot from hop/index {}/{} to {}/{}.",
                //     current_hop,
                //     current_idx,
                //     candidate_hop + victim_hop,
                //     victim_idx
                // );
                return Some(candidate_hop + victim_hop);
            }
        }
        None
    }

    #[cfg(test)]
    fn check_consistency(&self) -> bool {
        for (bucket_idx, bucket) in self.buckets.iter().enumerate() {
            for hop in (0..HOP_LIMIT).filter(|hop| bucket.hop_info.get(*hop)) {
                let entry_idx = (bucket_idx + hop) % self.capacity();
                if self.buckets[entry_idx].is_vacant() {
                    println!(
                        "Bucket {} says hop {} / entry {} is occupied, but it's vacant.",
                        bucket_idx, hop, entry_idx,
                    );
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::collections::HashMap;

    #[test]
    fn test_insertion() {
        let mut map: HsHashMap<i32, i32> = HsHashMap::new();
        assert_eq!(map.is_empty(), true);
        assert_eq!(map.len(), 0);
        for key in 0..(HOP_LIMIT * 16) as i32 {
            assert_eq!(map.get(&key), None);
            println!("insert({}, {})", key, key);
            assert_eq!(map.insert(key, key), None);
            assert_eq!(map.is_empty(), false);
            assert_eq!(map.len(), key as usize + 1);
            assert_eq!(*map.get(&key).unwrap(), key);
            assert_eq!(*map.get_mut(&key).unwrap(), key);
            assert!(map.check_consistency());
        }
        for val in 2..10 {
            println!("insert(1, {})", val);
            assert_eq!(map.insert(1, val), Some(val - 1));
            assert!(map.check_consistency());
        }
        assert_eq!(map.is_empty(), false);
    }

    // TODO: Add an AssociativeArray trait and make the benchmarks generic.

    const MAP_SIZE: usize = 100000;

    #[bench]
    fn bench_hopscotch_insertion(b: &mut test::Bencher) {
        let mut map: HsHashMap<i32, i32> = HsHashMap::new();
        let mut key: i32 = 0;
        b.iter(|| {
            map.insert(key, key);
            key += 1;
            if key as usize == MAP_SIZE {
                map.clear();
                key = 0;
            }
        })
    }

    #[bench]
    fn bench_hopscotch_get(b: &mut test::Bencher) {
        let mut map: HsHashMap<i32, i32> = HsHashMap::new();
        for key in 0..MAP_SIZE as i32 {
            map.insert(key * 2, key * 2);
        }
        let mut key: i32 = 0;
        b.iter(|| {
            map.get(&key);
            key += 1;
            if key as usize == MAP_SIZE * 2 {
                key = 0;
            }
        })
    }

    #[bench]
    fn bench_hashmap_insertion(b: &mut test::Bencher) {
        let mut map: HashMap<i32, i32> = HashMap::new();
        let mut key: i32 = 0;
        b.iter(|| {
            map.insert(key, key);
            key += 1;
            if key as usize == MAP_SIZE {
                map.clear();
                key = 0;
            }
        })
    }

    #[bench]
    fn bench_hashmap_get(b: &mut test::Bencher) {
        let mut map: HashMap<i32, i32> = HashMap::new();
        for key in 0..MAP_SIZE as i32 {
            map.insert(key * 2, key * 2);
        }
        let mut key: i32 = 0;
        b.iter(|| {
            map.get(&key);
            key += 1;
            if key as usize == MAP_SIZE * 2 {
                key = 0;
            }
        })
    }

    #[bench]
    fn bench_hashmap_clone(b: &mut test::Bencher) {
        let mut map: HashMap<i32, i32> = HashMap::new();
        for key in 0..MAP_SIZE as i32 {
            map.insert(key, key);
        }
        let mut map_copy = map.clone();
        b.iter(|| {
            map_copy = map.clone();
        })
    }

    #[bench]
    fn bench_hashmap_removal(b: &mut test::Bencher) {
        // NOTE: This measures removal plus an amortized clone.
        // Consider switching to Criterion, which has better support for
        // operations that need fresh input. See:
        // https://bheisler.github.io/criterion.rs/criterion/struct.Bencher
        let mut map: HashMap<i32, i32> = HashMap::new();
        for key in 0..MAP_SIZE as i32 {
            map.insert(key, key);
        }
        let mut key: i32 = 0;
        let mut map_copy = map.clone();
        b.iter(|| {
            map_copy.remove(&key);
            key += 1;
            if key as usize == MAP_SIZE {
                map_copy = map.clone();
                key = 0;
            }
        })
    }

    #[bench]
    fn bench_btreemap_insertion(b: &mut test::Bencher) {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        let mut key: i32 = 0;
        b.iter(|| {
            map.insert(key, key);
            key += 1;
            if key as usize == MAP_SIZE {
                map.clear();
                key = 0;
            }
        })
    }

    #[bench]
    fn bench_btreemap_get(b: &mut test::Bencher) {
        let mut map: BTreeMap<i32, i32> = BTreeMap::new();
        for key in 0..MAP_SIZE as i32 {
            map.insert(key * 2, key * 2);
        }
        let mut key: i32 = 0;
        b.iter(|| {
            map.get(&key);
            key += 1;
            if key as usize == MAP_SIZE * 2 {
                key = 0;
            }
        })
    }
}
