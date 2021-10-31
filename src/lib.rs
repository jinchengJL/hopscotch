use bitmaps::Bitmap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem;

const HOP_RANGE: usize = 32;

#[derive(Clone, Debug)]
struct Entry<K, V>
where
    K: Hash + Eq,
{
    key: K,
    value: V,
}

// A physical hash slot storing an entry.
#[derive(Clone, Debug)]
struct Slot<K, V>
where
    K: Hash + Eq,
{
    hop_info: Bitmap<HOP_RANGE>,
    entry: Option<Entry<K, V>>,
}

// A logical hash bucket storing up to HOP_RANGE entries.
// TODO: Add a method for iterating over occupied hops.
trait Bucket {
    fn has_hop(&self, hop: usize) -> bool;

    fn set_hop(&mut self, hop: usize, occupied: bool);

    fn is_full(&self) -> bool;
}

#[derive(Clone, Debug)]
pub struct HsHashMap<K, V>
where
    K: Hash + Eq,
{
    slots: Vec<Slot<K, V>>,
    length: usize,
}

impl<K, V> Slot<K, V>
where
    K: Hash + Eq,
{
    fn new() -> Slot<K, V> {
        Slot {
            hop_info: Bitmap::new(),
            entry: None,
        }
    }

    fn is_vacant(&self) -> bool {
        self.entry.is_none()
    }
}

impl<K, V> Bucket for Slot<K, V>
where
    K: Hash + Eq,
{
    fn has_hop(&self, hop: usize) -> bool {
        self.hop_info.get(hop)
    }

    fn set_hop(&mut self, hop: usize, occupied: bool) {
        self.hop_info.set(hop, occupied);
    }

    fn is_full(&self) -> bool {
        self.hop_info.is_full()
    }
}

// TODO: Iterators.
// TODO: Parameterize the hash function.
// TODO: Parameterize the hop range.
impl<K, V> HsHashMap<K, V>
where
    K: Hash + Eq,
{
    // Public functions.
    pub fn new() -> HsHashMap<K, V> {
        let mut slots: Vec<Slot<K, V>> = vec![];
        slots.resize_with(HOP_RANGE, Slot::new);
        HsHashMap {
            slots: slots,
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
        if self.slots[bucket_idx].is_full() {
            self.rehash(self.capacity() * 2);
            return self.insert(k, v);
        }
        let capacity = self.capacity();
        let vacant_hop = (0..capacity).find(|hop| {
            let entry_idx = self.get_idx(bucket_idx, *hop);
            self.slots[entry_idx].entry.is_none()
        });
        if vacant_hop.is_none() {
            self.rehash(self.capacity() * 2);
            return self.insert(k, v);
        }
        // Keep moving the empty slot closer to bucket_idx.
        let mut hop = vacant_hop.unwrap();
        while hop >= HOP_RANGE {
            hop = match self.move_closer(bucket_idx, hop) {
                None => {
                    self.rehash(self.capacity() * 2);
                    return self.insert(k, v);
                }
                Some(new_hop) => new_hop,
            }
        }
        let entry_idx = self.get_idx(bucket_idx, hop);
        let slots = &mut self.slots;
        debug_assert!(slots[entry_idx].is_vacant());
        debug_assert!(!slots[bucket_idx].has_hop(hop));
        slots[bucket_idx].set_hop(hop, true);
        slots[entry_idx].entry = Some(Entry { key: k, value: v });
        self.length += 1;
        None
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        let bucket_idx = self.get_bucket(&k);
        let hop_info = &self.slots[bucket_idx].hop_info;
        for hop in (0..HOP_RANGE).filter(|hop| hop_info.get(*hop)) {
            let entry_idx = self.get_idx(bucket_idx, hop);
            match &self.slots[entry_idx].entry {
                None => {
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

    pub fn contains_key(&self, k: &K) -> bool {
        self.get(k).is_some()
    }

    pub fn remove(&mut self, k: &K) -> Option<V> {
        let bucket_idx = self.get_bucket(&k);
        let hop_info = &self.slots[bucket_idx].hop_info;
        for hop in (0..HOP_RANGE).filter(|hop| hop_info.get(*hop)) {
            let entry_idx = self.get_idx(bucket_idx, hop);
            match &self.slots[entry_idx].entry {
                None => {
                    std::debug_assert!(false, "Bucket should not be empty.");
                    continue;
                }
                Some(Entry { key, value: _ }) if key == k => {
                    self.slots[bucket_idx].set_hop(hop, false);
                    self.length -= 1;
                    return Some(mem::take(&mut self.slots[entry_idx].entry).unwrap().value);
                }
                _ => {}
            }
        }
        None
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.slots.resize_with(HOP_RANGE, Slot::new);
        self.length = 0;
    }

    // Private functions.
    fn get_bucket(&self, k: &K) -> usize {
        (self.get_hash(&k) as usize) & (self.capacity() - 1)
    }

    fn get_hash(&self, k: &K) -> u64 {
        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        hasher.finish()
    }

    fn get_idx(&self, base: usize, hop: usize) -> usize {
        (base + hop) & (self.capacity() - 1)
    }

    fn rehash(&mut self, new_capacity: usize) {
        let slots = mem::replace(&mut self.slots, vec![]);
        let original_length = self.length;
        self.slots.resize_with(new_capacity, Slot::new);
        self.length = 0;
        for slot in slots {
            if let Some(entry) = slot.entry {
                self.insert(entry.key, entry.value);
            }
        }
        debug_assert_eq!(self.length, original_length);
    }

    // Moves the vacant slot at `hop` closer to `origin_idx`.
    fn move_closer(&mut self, origin_idx: usize, vacant_hop: usize) -> Option<usize> {
        debug_assert!(vacant_hop >= HOP_RANGE);
        let vacant_idx = self.get_idx(origin_idx, vacant_hop);
        debug_assert!(self.slots[vacant_idx].is_vacant());
        for candidate_hop in (vacant_hop - HOP_RANGE + 1)..vacant_hop {
            let candidate_idx = self.get_idx(origin_idx, candidate_hop);
            let candidate_to_vacant = vacant_hop - candidate_hop;
            // Hop info for vacant_hop must be empty.
            debug_assert!(candidate_to_vacant < HOP_RANGE);
            debug_assert!(!self.slots[candidate_idx].has_hop(candidate_to_vacant));
            // Find the earliest slot for `candidate_idx` that isn't empty.
            let victim_hop = match self.slots[candidate_idx].hop_info.first_index() {
                Some(hop) if hop < candidate_to_vacant => hop,
                _ => {
                    continue;
                }
            };
            // Swap the victim entry with the empty slot.
            let victim_idx = self.get_idx(candidate_idx, victim_hop);
            self.slots[vacant_idx].entry = mem::take(&mut self.slots[victim_idx].entry);
            debug_assert!(self.slots[victim_idx].is_vacant());
            self.slots[candidate_idx].set_hop(victim_hop, false);
            self.slots[candidate_idx].set_hop(candidate_to_vacant, true);
            return Some(candidate_hop + victim_hop);
        }
        None
    }

    #[cfg(test)]
    fn check_consistency(&self) -> bool {
        for (bucket_idx, bucket) in self.slots.iter().enumerate() {
            for hop in (0..HOP_RANGE).filter(|hop| bucket.hop_info.get(*hop)) {
                let entry_idx = self.get_idx(bucket_idx, hop);
                if self.slots[entry_idx].is_vacant() {
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

    #[test]
    fn basic_test() {
        let mut map: HsHashMap<i32, i32> = HsHashMap::new();
        assert_eq!(map.is_empty(), true);
        assert_eq!(map.len(), 0);
        for key in 0..(HOP_RANGE * 16) as i32 {
            assert_eq!(map.get(&key), None);
            assert_eq!(map.insert(key, key), None);
            assert_eq!(map.is_empty(), false);
            assert_eq!(map.len(), key as usize + 1);
            assert_eq!(*map.get(&key).unwrap(), key);
            assert_eq!(*map.get_mut(&key).unwrap(), key);
        }
        assert!(map.check_consistency());
        for val in 1..10 {
            assert_eq!(map.insert(0, val), Some(val - 1));
        }
        assert!(map.check_consistency());
        for val in 2..10 {
            let slot = map.get_mut(&1).unwrap();
            *slot = val;
            assert_eq!(*map.get(&1).unwrap(), val);
        }
        assert!(map.check_consistency());
        assert_eq!(map.len(), HOP_RANGE * 16);
        for key in 2..(HOP_RANGE * 8) as i32 {
            assert_eq!(*map.get(&key).unwrap(), key);
            assert_eq!(map.remove(&key), Some(key), "{:?}", key);
        }
        assert!(map.check_consistency());
        assert_eq!(map.len(), HOP_RANGE * 8 + 2);
        assert_eq!(map.is_empty(), false);
    }
}
