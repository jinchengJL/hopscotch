use bitmaps::Bitmap;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::iter::Extend;
use std::iter::FromIterator;
use std::iter::FusedIterator;
use std::mem;
use std::ops::Index;

const DEFAULT_HOP_RANGE: usize = 32;

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
    hop_info: Bitmap<DEFAULT_HOP_RANGE>,
    entry: Option<Entry<K, V>>,
}

// A logical hash bucket storing up to DEFAULT_HOP_RANGE entries.
// TODO: Add an iterator over occupied hops.
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
    hop_range: usize,
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

pub struct Iter<'a, K, V>
where
    K: Hash + Eq,
{
    map: &'a HsHashMap<K, V>,
    index: usize,
    remaining_items: usize,
}

impl<'a, K: Hash + Eq, V> Iter<'a, K, V> {
    fn new(map: &'a HsHashMap<K, V>) -> Self {
        Self {
            map: map,
            index: 0,
            remaining_items: map.len(),
        }
    }
}

impl<'a, K: Hash + Eq, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.map.capacity() && self.map.slots[self.index].is_vacant() {
            self.index += 1;
        }
        if self.index == self.map.capacity() {
            return None;
        }
        let entry = &self.map.slots[self.index].entry.as_ref()?;
        self.index += 1;
        self.remaining_items -= 1;
        Some((&entry.key, &entry.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining_items, Some(self.remaining_items))
    }
}

impl<'a, K: Hash + Eq, V> ExactSizeIterator for Iter<'a, K, V> {}
impl<'a, K: Hash + Eq, V> FusedIterator for Iter<'a, K, V> {}

pub struct IterMut<'a, K, V>
where
    K: Hash + Eq,
{
    map: &'a mut HsHashMap<K, V>,
    index: usize,
    remaining_items: usize,
}

impl<'a, K: Hash + Eq, V> IterMut<'a, K, V> {
    fn new(map: &'a mut HsHashMap<K, V>) -> Self {
        let len = map.len();
        Self {
            map: map,
            index: 0,
            remaining_items: len,
        }
    }
}

impl<'a, K: Hash + Eq, V> Iterator for IterMut<'a, K, V>
where
    K: Hash + Eq,
{
    type Item = (&'a mut K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.map.capacity() && self.map.slots[self.index].is_vacant() {
            self.index += 1;
        }
        if self.index == self.map.capacity() {
            return None;
        }
        let index = self.index;
        self.index += 1;
        self.remaining_items -= 1;
        // NOTE: Due to lifetime issues, unsafe code is often required to
        // implement mutable iterators.
        unsafe {
            let slot_ptr = self.map.slots.as_mut_ptr().add(index);
            let slot = &mut *slot_ptr;
            let entry = slot.entry.as_mut()?;
            Some((&mut entry.key, &mut entry.value))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining_items, Some(self.remaining_items))
    }
}

impl<'a, K: Hash + Eq, V> ExactSizeIterator for IterMut<'a, K, V> {}
impl<'a, K: Hash + Eq, V> FusedIterator for IterMut<'a, K, V> {}

pub struct IntoIter<K, V>
where
    K: Hash + Eq,
{
    map: HsHashMap<K, V>,
    index: usize,
    remaining_items: usize,
}

impl<K: Hash + Eq, V> IntoIter<K, V> {
    fn new(map: HsHashMap<K, V>) -> Self {
        let len = map.len();
        Self {
            map: map,
            index: 0,
            remaining_items: len,
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V>
where
    K: Hash + Eq,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.map.capacity() && self.map.slots[self.index].is_vacant() {
            self.index += 1;
        }
        if self.index == self.map.capacity() {
            return None;
        }
        let entry = mem::take(&mut self.map.slots[self.index].entry)?;
        self.index += 1;
        self.remaining_items -= 1;
        Some((entry.key, entry.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining_items, Some(self.remaining_items))
    }
}

impl<K: Hash + Eq, V> ExactSizeIterator for IntoIter<K, V> {}
impl<K: Hash + Eq, V> FusedIterator for IntoIter<K, V> {}

impl<'a, K: Hash + Eq, V> IntoIterator for &'a HsHashMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<'a, K: Hash + Eq, V> IntoIterator for &'a mut HsHashMap<K, V> {
    type Item = (&'a mut K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        IterMut::new(self)
    }
}

impl<K: Hash + Eq, V> IntoIterator for HsHashMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<K: Hash + Eq, V> Extend<(K, V)> for HsHashMap<K, V> {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (K, V)>,
    {
        // TODO: Call reserve() first.
        for (k, v) in iter.into_iter() {
            self.insert(k, v);
        }
    }
}

impl<'a, K, V> Extend<(&'a K, &'a V)> for HsHashMap<K, V>
where
    K: Eq + Hash + Copy,
    V: Copy,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = (&'a K, &'a V)>,
    {
        self.extend(iter.into_iter().map(|(&k, &v)| (k, v)));
    }
}

impl<K: Hash + Eq, V> FromIterator<(K, V)> for HsHashMap<K, V> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (K, V)>,
    {
        let mut map = HsHashMap::new();
        map.extend(iter);
        map
    }
}

pub struct Keys<'a, K: Hash + Eq, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K: Hash + Eq, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub struct IntoKeys<K: Hash + Eq, V> {
    inner: IntoIter<K, V>,
}

impl<'a, K: Hash + Eq, V> Iterator for IntoKeys<K, V> {
    type Item = K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub struct Values<'a, K: Hash + Eq, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K: Hash + Eq, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub struct ValuesMut<'a, K: Hash + Eq, V> {
    inner: IterMut<'a, K, V>,
}

impl<'a, K: Hash + Eq, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub struct IntoValues<K: Hash + Eq, V> {
    inner: IntoIter<K, V>,
}

impl<'a, K: Hash + Eq, V> Iterator for IntoValues<K, V> {
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> Index<&K> for HsHashMap<K, V>
where
    K: Hash + Eq,
{
    type Output = V;
    fn index(&self, key: &K) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

// TODO: Add missing features:
// - Custom hash function.
// - Zero capacity.
// - Manipulating the raw entry.
// - Equality.
impl<K, V> HsHashMap<K, V>
where
    K: Hash + Eq,
{
    // Public functions.
    pub fn new() -> HsHashMap<K, V> {
        Self::with_hop_range(DEFAULT_HOP_RANGE)
    }

    pub fn with_hop_range(hop_range: usize) -> HsHashMap<K, V> {
        let mut slots: Vec<Slot<K, V>> = vec![];
        slots.resize_with(hop_range, Slot::new);
        HsHashMap {
            slots: slots,
            length: 0,
            hop_range: hop_range,
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
        while hop >= self.hop_range {
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
        for hop in (0..self.hop_range).filter(|hop| hop_info.get(*hop)) {
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
        match self.remove_entry(k) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    pub fn remove_entry(&mut self, k: &K) -> Option<(K, V)> {
        let bucket_idx = self.get_bucket(&k);
        let hop_info = &self.slots[bucket_idx].hop_info;
        for hop in (0..self.hop_range).filter(|hop| hop_info.get(*hop)) {
            let entry_idx = self.get_idx(bucket_idx, hop);
            match &self.slots[entry_idx].entry {
                None => {
                    std::debug_assert!(false, "Bucket should not be empty.");
                    continue;
                }
                Some(Entry { key, value: _ }) if key == k => {
                    self.slots[bucket_idx].set_hop(hop, false);
                    self.length -= 1;
                    let entry = mem::take(&mut self.slots[entry_idx].entry).unwrap();
                    return Some((entry.key, entry.value));
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
        self.slots.resize_with(self.hop_range, Slot::new);
        self.length = 0;
    }

    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(&self)
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut::new(self)
    }

    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    pub fn into_keys(self) -> IntoKeys<K, V> {
        IntoKeys {
            inner: self.into_iter(),
        }
    }

    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    pub fn into_values(self) -> IntoValues<K, V> {
        IntoValues {
            inner: self.into_iter(),
        }
    }

    // Private functions.
    fn get_bucket(&self, k: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        let hash = hasher.finish();
        (hash as usize) & (self.capacity() - 1)
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

    // Moves the vacant slot at `origin_idx + vacant_hop` closer to `origin_idx`.
    fn move_closer(&mut self, origin_idx: usize, vacant_hop: usize) -> Option<usize> {
        debug_assert!(vacant_hop >= self.hop_range);
        let vacant_idx = self.get_idx(origin_idx, vacant_hop);
        debug_assert!(self.slots[vacant_idx].is_vacant());
        for candidate_hop in (vacant_hop - self.hop_range + 1)..vacant_hop {
            let candidate_idx = self.get_idx(origin_idx, candidate_hop);
            let candidate_to_vacant = vacant_hop - candidate_hop;
            // Hop info for vacant_hop must be empty.
            debug_assert!(candidate_to_vacant < self.hop_range);
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
            for hop in (0..self.hop_range).filter(|hop| bucket.hop_info.get(*hop)) {
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
    use std::cell::RefCell;

    #[test]
    fn test_basic() {
        let mut map: HsHashMap<i32, i32> = HsHashMap::new();
        assert_eq!(map.is_empty(), true);
        assert_eq!(map.len(), 0);
        for key in 0..(DEFAULT_HOP_RANGE * 16) as i32 {
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
        assert_eq!(map.len(), DEFAULT_HOP_RANGE * 16);
        for key in 2..(DEFAULT_HOP_RANGE * 8) as i32 {
            assert_eq!(*map.get(&key).unwrap(), key);
            assert_eq!(map.remove(&key), Some(key), "{:?}", key);
        }
        assert!(map.check_consistency());
        assert_eq!(map.len(), DEFAULT_HOP_RANGE * 8 + 2);
        assert_eq!(map.is_empty(), false);
    }

    #[test]
    fn test_default_capacities() {
        type HM = HsHashMap<i32, i32>;

        let m = HM::new();
        assert_eq!(m.capacity(), DEFAULT_HOP_RANGE);

        // let m = HM::default();
        // assert_eq!(m.capacity(), DEFAULT_HOP_RANGE);

        // let m = HM::with_hasher(DefaultHashBuilder::default());
        // assert_eq!(m.capacity(), DEFAULT_HOP_RANGE);

        // let m = HM::with_capacity(0);
        // assert_eq!(m.capacity(), DEFAULT_HOP_RANGE);

        // let m = HM::with_capacity_and_hasher(0, DefaultHashBuilder::default());
        // assert_eq!(m.capacity(), DEFAULT_HOP_RANGE);

        // let mut m = HM::new();
        // m.insert(1, 1);
        // m.insert(2, 2);
        // m.remove(&1);
        // m.remove(&2);
        // m.shrink_to_fit();
        // assert_eq!(m.capacity(), DEFAULT_HOP_RANGE);

        // let mut m = HM::new();
        // m.reserve(0);
        // assert_eq!(m.capacity(), DEFAULT_HOP_RANGE);
    }

    // #[test]
    // fn test_create_capacity_zero() {
    //     let mut m = HsHashMap::with_capacity(0);

    //     assert!(m.insert(1, 1).is_none());

    //     assert!(m.contains_key(&1));
    //     assert!(!m.contains_key(&0));
    // }

    #[test]
    fn test_insert() {
        let mut m = HsHashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&2).unwrap(), 4);
    }

    #[test]
    fn test_clone() {
        let mut m = HsHashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        #[allow(clippy::redundant_clone)]
        let m2 = m.clone();
        assert_eq!(*m2.get(&1).unwrap(), 2);
        assert_eq!(*m2.get(&2).unwrap(), 4);
        assert_eq!(m2.len(), 2);
    }

    #[test]
    fn test_clone_from() {
        let mut m = HsHashMap::new();
        let mut m2 = HsHashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        m2.clone_from(&m);
        assert_eq!(*m2.get(&1).unwrap(), 2);
        assert_eq!(*m2.get(&2).unwrap(), 4);
        assert_eq!(m2.len(), 2);
    }

    thread_local! { static DROP_VECTOR: RefCell<Vec<i32>> = RefCell::new(Vec::new()) }

    #[derive(Hash, PartialEq, Eq)]
    struct Droppable {
        k: usize,
    }

    impl Droppable {
        fn new(k: usize) -> Droppable {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[k] += 1;
            });

            Droppable { k }
        }
    }

    impl Drop for Droppable {
        fn drop(&mut self) {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[self.k] -= 1;
            });
        }
    }

    impl Clone for Droppable {
        fn clone(&self) -> Self {
            Droppable::new(self.k)
        }
    }

    #[test]
    fn test_drops() {
        DROP_VECTOR.with(|slot| {
            *slot.borrow_mut() = vec![0; 200];
        });

        {
            let mut m = HsHashMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..100 {
                let d1 = Droppable::new(i);
                let d2 = Droppable::new(i + 100);
                m.insert(d1, d2);
            }

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for i in 0..50 {
                let k = Droppable::new(i);
                let v = m.remove(&k);

                assert!(v.is_some());

                DROP_VECTOR.with(|v| {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i + 100], 1);
                });
            }

            DROP_VECTOR.with(|v| {
                for i in 0..50 {
                    assert_eq!(v.borrow()[i], 0);
                    assert_eq!(v.borrow()[i + 100], 0);
                }

                for i in 50..100 {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i + 100], 1);
                }
            });
        }

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_into_iter_drops() {
        DROP_VECTOR.with(|v| {
            *v.borrow_mut() = vec![0; 200];
        });

        let hm = {
            let mut hm = HsHashMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..100 {
                let d1 = Droppable::new(i);
                let d2 = Droppable::new(i + 100);
                hm.insert(d1, d2);
            }

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            hm
        };

        // By the way, ensure that cloning doesn't screw up the dropping.
        drop(hm.clone());

        {
            let mut half = hm.into_iter().take(50);

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            #[allow(clippy::let_underscore_drop)] // kind-of a false positive
            for _ in half.by_ref() {}

            DROP_VECTOR.with(|v| {
                let nk = (0..100).filter(|&i| v.borrow()[i] == 1).count();

                let nv = (0..100).filter(|&i| v.borrow()[i + 100] == 1).count();

                assert_eq!(nk, 50);
                assert_eq!(nv, 50);
            });
        };

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_empty_remove() {
        let mut m: HsHashMap<i32, bool> = HsHashMap::new();
        assert_eq!(m.remove(&0), None);
    }

    // #[test]
    // fn test_empty_entry() {
    //     let mut m: HsHashMap<i32, bool> = HsHashMap::new();
    //     match m.entry(0) {
    //         Occupied(_) => panic!(),
    //         Vacant(_) => {}
    //     }
    //     assert!(*m.entry(0).or_insert(true));
    //     assert_eq!(m.len(), 1);
    // }

    #[test]
    fn test_empty_iter() {
        let mut m: HsHashMap<i32, bool> = HsHashMap::new();
        //     assert_eq!(m.drain().next(), None);
        assert_eq!(m.keys().next(), None);
        assert_eq!(m.values().next(), None);
        assert_eq!(m.values_mut().next(), None);
        assert_eq!(m.iter().next(), None);
        assert_eq!(m.iter_mut().next(), None);
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert_eq!(m.into_iter().next(), None);
    }

    #[test]
    fn test_lots_of_insertions() {
        let loop_end = 257;
        let mut m = HsHashMap::new();

        // Try this a few times to make sure we never screw up the hashmap's
        // internal state.
        for _ in 0..10 {
            assert!(m.is_empty());

            for i in 1..loop_end {
                assert!(m.insert(i, i).is_none());

                for j in 1..=i {
                    let r = m.get(&j);
                    assert_eq!(r, Some(&j));
                }

                for j in i + 1..loop_end {
                    let r = m.get(&j);
                    assert_eq!(r, None);
                }
            }

            for i in loop_end..(loop_end * 2 - 1) {
                assert!(!m.contains_key(&i));
            }

            // remove forwards
            for i in 1..loop_end {
                assert!(m.remove(&i).is_some());

                for j in 1..=i {
                    assert!(!m.contains_key(&j));
                }

                for j in i + 1..loop_end {
                    assert!(m.contains_key(&j));
                }
            }

            for i in 1..loop_end {
                assert!(!m.contains_key(&i));
            }

            for i in 1..loop_end {
                assert!(m.insert(i, i).is_none());
            }

            // remove backwards
            for i in (1..loop_end).rev() {
                assert!(m.remove(&i).is_some());

                for j in i..loop_end {
                    assert!(!m.contains_key(&j));
                }

                for j in 1..i {
                    assert!(m.contains_key(&j));
                }
            }
        }
    }

    #[test]
    fn test_find_mut() {
        let mut m = HsHashMap::new();
        assert!(m.insert(1, 12).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.insert(5, 14).is_none());
        let new = 100;
        match m.get_mut(&5) {
            None => panic!(),
            Some(x) => *x = new,
        }
        assert_eq!(m.get(&5), Some(&new));
    }

    #[test]
    fn test_insert_overwrite() {
        let mut m = HsHashMap::new();
        assert!(m.insert(1, 2).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert!(!m.insert(1, 3).is_none());
        assert_eq!(*m.get(&1).unwrap(), 3);
    }

    #[test]
    fn test_is_empty() {
        let mut m = HsHashMap::new();
        assert!(m.insert(1, 2).is_none());
        assert!(!m.is_empty());
        assert!(m.remove(&1).is_some());
        assert!(m.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut m = HsHashMap::new();
        m.insert(1, 2);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_remove_entry() {
        let mut m = HsHashMap::new();
        m.insert(1, 2);
        assert_eq!(m.remove_entry(&1), Some((1, 2)));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_iterate() {
        let mut m = HsHashMap::new();
        for i in 0..32 {
            assert!(m.insert(i, i * 2).is_none());
        }
        assert_eq!(m.len(), 32);

        let mut observed: u32 = 0;
        for (k, v) in &m {
            assert_eq!(*v, *k * 2);
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);

        observed = 0;
        for (k, v) in &mut m {
            assert_eq!(*v, *k * 2);
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);

        observed = 0;
        for (k, v) in m {
            assert_eq!(v, k * 2);
            observed |= 1 << k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: HsHashMap<_, _> = vec.into_iter().collect();
        let keys: Vec<_> = map.keys().copied().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: HsHashMap<_, _> = vec.into_iter().collect();
        let values: Vec<_> = map.values().copied().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_values_mut() {
        let vec = vec![(1, 1), (2, 2), (3, 3)];
        let mut map: HsHashMap<_, _> = vec.into_iter().collect();
        for value in map.values_mut() {
            *value *= 2;
        }
        let values: Vec<_> = map.values().copied().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&2));
        assert!(values.contains(&4));
        assert!(values.contains(&6));
    }

    #[test]
    fn test_into_keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: HsHashMap<_, _> = vec.into_iter().collect();
        let keys: Vec<_> = map.into_keys().collect();

        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_into_values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: HsHashMap<_, _> = vec.into_iter().collect();
        let values: Vec<_> = map.into_values().collect();

        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_find() {
        let mut m = HsHashMap::new();
        assert!(m.get(&1).is_none());
        m.insert(1, 2);
        match m.get(&1) {
            None => panic!(),
            Some(v) => assert_eq!(*v, 2),
        }
    }

    // #[test]
    // fn test_eq() {
    //     let mut m1 = HsHashMap::new();
    //     m1.insert(1, 2);
    //     m1.insert(2, 3);
    //     m1.insert(3, 4);

    //     let mut m2 = HsHashMap::new();
    //     m2.insert(1, 2);
    //     m2.insert(2, 3);

    //     assert!(m1 != m2);

    //     m2.insert(3, 4);

    //     assert_eq!(m1, m2);
    // }

    // #[test]
    // fn test_show() {
    //     let mut map = HsHashMap::new();
    //     let empty: HsHashMap<i32, i32> = HsHashMap::new();

    //     map.insert(1, 2);
    //     map.insert(3, 4);

    //     let map_str = format!("{:?}", map);

    //     assert!(map_str == "{1: 2, 3: 4}" || map_str == "{3: 4, 1: 2}");
    //     assert_eq!(format!("{:?}", empty), "{}");
    // }

    #[test]
    fn test_expand() {
        let mut m = HsHashMap::new();

        assert_eq!(m.len(), 0);
        assert!(m.is_empty());

        let mut i = 0;
        let old_cap = m.capacity();
        while old_cap == m.capacity() {
            m.insert(i, i);
            i += 1;
        }

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
    }

    // #[test]
    // fn test_behavior_resize_policy() {
    //     let mut m = HsHashMap::new();

    //     assert_eq!(m.len(), 0);
    //     assert_eq!(m.raw_capacity(), 1);
    //     assert!(m.is_empty());

    //     m.insert(0, 0);
    //     m.remove(&0);
    //     assert!(m.is_empty());
    //     let initial_raw_cap = m.raw_capacity();
    //     m.reserve(initial_raw_cap);
    //     let raw_cap = m.raw_capacity();

    //     assert_eq!(raw_cap, initial_raw_cap * 2);

    //     let mut i = 0;
    //     for _ in 0..raw_cap * 3 / 4 {
    //         m.insert(i, i);
    //         i += 1;
    //     }
    //     // three quarters full

    //     assert_eq!(m.len(), i);
    //     assert_eq!(m.raw_capacity(), raw_cap);

    //     for _ in 0..raw_cap / 4 {
    //         m.insert(i, i);
    //         i += 1;
    //     }
    //     // half full

    //     let new_raw_cap = m.raw_capacity();
    //     assert_eq!(new_raw_cap, raw_cap * 2);

    //     for _ in 0..raw_cap / 2 - 1 {
    //         i -= 1;
    //         m.remove(&i);
    //         assert_eq!(m.raw_capacity(), new_raw_cap);
    //     }
    //     // A little more than one quarter full.
    //     m.shrink_to_fit();
    //     assert_eq!(m.raw_capacity(), raw_cap);
    //     // again, a little more than half full
    //     for _ in 0..raw_cap / 2 {
    //         i -= 1;
    //         m.remove(&i);
    //     }
    //     m.shrink_to_fit();

    //     assert_eq!(m.len(), i);
    //     assert!(!m.is_empty());
    //     assert_eq!(m.raw_capacity(), initial_raw_cap);
    // }

    // #[test]
    // fn test_reserve_shrink_to_fit() {
    //     let mut m = HsHashMap::new();
    //     m.insert(0, 0);
    //     m.remove(&0);
    //     assert!(m.capacity() >= m.len());
    //     for i in 0..128 {
    //         m.insert(i, i);
    //     }
    //     m.reserve(256);

    //     let usable_cap = m.capacity();
    //     for i in 128..(128 + 256) {
    //         m.insert(i, i);
    //         assert_eq!(m.capacity(), usable_cap);
    //     }

    //     for i in 100..(128 + 256) {
    //         assert_eq!(m.remove(&i), Some(i));
    //     }
    //     m.shrink_to_fit();

    //     assert_eq!(m.len(), 100);
    //     assert!(!m.is_empty());
    //     assert!(m.capacity() >= m.len());

    //     for i in 0..100 {
    //         assert_eq!(m.remove(&i), Some(i));
    //     }
    //     m.shrink_to_fit();
    //     m.insert(0, 0);

    //     assert_eq!(m.len(), 1);
    //     assert!(m.capacity() >= m.len());
    //     assert_eq!(m.remove(&0), Some(0));
    // }

    #[test]
    fn test_from_iter() {
        let xs = [(1, 1), (2, 2), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HsHashMap<_, _> = xs.iter().copied().collect();

        for &(k, v) in &xs {
            assert_eq!(map.get(&k), Some(&v));
        }

        assert_eq!(map.iter().len(), xs.len() - 1);
    }

    #[test]
    fn test_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HsHashMap<_, _> = xs.iter().copied().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HsHashMap<_, _> = xs.iter().copied().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_mut_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: HsHashMap<_, _> = xs.iter().copied().collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_mut_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: HsHashMap<_, _> = xs.iter().copied().collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_index() {
        let mut map = HsHashMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[&2], 1);
    }

    #[test]
    #[should_panic]
    fn test_index_nonexistent() {
        let mut map = HsHashMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        #[allow(clippy::no_effect)] // false positive lint
        map[&4];
    }

    // #[test]
    // fn test_entry() {
    //     let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

    //     let mut map: HsHashMap<_, _> = xs.iter().copied().collect();

    //     // Existing key (insert)
    //     match map.entry(1) {
    //         Vacant(_) => unreachable!(),
    //         Occupied(mut view) => {
    //             assert_eq!(view.get(), &10);
    //             assert_eq!(view.insert(100), 10);
    //         }
    //     }
    //     assert_eq!(map.get(&1).unwrap(), &100);
    //     assert_eq!(map.len(), 6);

    //     // Existing key (update)
    //     match map.entry(2) {
    //         Vacant(_) => unreachable!(),
    //         Occupied(mut view) => {
    //             let v = view.get_mut();
    //             let new_v = (*v) * 10;
    //             *v = new_v;
    //         }
    //     }
    //     assert_eq!(map.get(&2).unwrap(), &200);
    //     assert_eq!(map.len(), 6);

    //     // Existing key (take)
    //     match map.entry(3) {
    //         Vacant(_) => unreachable!(),
    //         Occupied(view) => {
    //             assert_eq!(view.remove(), 30);
    //         }
    //     }
    //     assert_eq!(map.get(&3), None);
    //     assert_eq!(map.len(), 5);

    //     // Inexistent key (insert)
    //     match map.entry(10) {
    //         Occupied(_) => unreachable!(),
    //         Vacant(view) => {
    //             assert_eq!(*view.insert(1000), 1000);
    //         }
    //     }
    //     assert_eq!(map.get(&10).unwrap(), &1000);
    //     assert_eq!(map.len(), 6);
    // }

    // #[test]
    // fn test_entry_take_doesnt_corrupt() {
    //     #![allow(deprecated)] //rand
    //                           // Test for #19292
    //     fn check(m: &HsHashMap<i32, ()>) {
    //         for k in m.keys() {
    //             assert!(m.contains_key(k), "{} is in keys() but not in the map?", k);
    //         }
    //     }

    //     let mut m = HsHashMap::new();

    //     let mut rng = {
    //         let seed = u64::from_le_bytes(*b"testseed");
    //         SmallRng::seed_from_u64(seed)
    //     };

    //     // Populate the map with some items.
    //     for _ in 0..50 {
    //         let x = rng.gen_range(-10..10);
    //         m.insert(x, ());
    //     }

    //     for _ in 0..1000 {
    //         let x = rng.gen_range(-10..10);
    //         match m.entry(x) {
    //             Vacant(_) => {}
    //             Occupied(e) => {
    //                 e.remove();
    //             }
    //         }

    //         check(&m);
    //     }
    // }

    #[test]
    fn test_extend_ref() {
        let mut a = HsHashMap::new();
        a.insert(1, "one");
        let mut b = HsHashMap::new();
        b.insert(2, "two");
        b.insert(3, "three");

        a.extend(&b);

        assert_eq!(a.len(), 3);
        assert_eq!(a[&1], "one");
        assert_eq!(a[&2], "two");
        assert_eq!(a[&3], "three");
    }

    #[test]
    fn test_capacity_not_less_than_len() {
        let mut a = HsHashMap::new();
        let mut item = 0;

        for _ in 0..116 {
            a.insert(item, 0);
            item += 1;
        }

        assert!(a.capacity() > a.len());

        let free = a.capacity() - a.len();
        for _ in 0..free {
            a.insert(item, 0);
            item += 1;
        }

        assert_eq!(a.len(), a.capacity());

        // Insert at capacity should cause allocation.
        a.insert(item, 0);
        assert!(a.capacity() > a.len());
    }

    // #[test]
    // fn test_occupied_entry_key() {
    //     let mut a = HsHashMap::new();
    //     let key = "hello there";
    //     let value = "value goes here";
    //     assert!(a.is_empty());
    //     a.insert(key, value);
    //     assert_eq!(a.len(), 1);
    //     assert_eq!(a[key], value);

    //     match a.entry(key) {
    //         Vacant(_) => panic!(),
    //         Occupied(e) => assert_eq!(key, *e.key()),
    //     }
    //     assert_eq!(a.len(), 1);
    //     assert_eq!(a[key], value);
    // }

    // #[test]
    // fn test_vacant_entry_key() {
    //     let mut a = HsHashMap::new();
    //     let key = "hello there";
    //     let value = "value goes here";

    //     assert!(a.is_empty());
    //     match a.entry(key) {
    //         Occupied(_) => panic!(),
    //         Vacant(e) => {
    //             assert_eq!(key, *e.key());
    //             e.insert(value);
    //         }
    //     }
    //     assert_eq!(a.len(), 1);
    //     assert_eq!(a[key], value);
    // }

    // #[test]
    // fn test_occupied_entry_replace_entry_with() {
    //     let mut a = HsHashMap::new();

    //     let key = "a key";
    //     let value = "an initial value";
    //     let new_value = "a new value";

    //     let entry = a.entry(key).insert(value).replace_entry_with(|k, v| {
    //         assert_eq!(k, &key);
    //         assert_eq!(v, value);
    //         Some(new_value)
    //     });

    //     match entry {
    //         Occupied(e) => {
    //             assert_eq!(e.key(), &key);
    //             assert_eq!(e.get(), &new_value);
    //         }
    //         Vacant(_) => panic!(),
    //     }

    //     assert_eq!(a[key], new_value);
    //     assert_eq!(a.len(), 1);

    //     let entry = match a.entry(key) {
    //         Occupied(e) => e.replace_entry_with(|k, v| {
    //             assert_eq!(k, &key);
    //             assert_eq!(v, new_value);
    //             None
    //         }),
    //         Vacant(_) => panic!(),
    //     };

    //     match entry {
    //         Vacant(e) => assert_eq!(e.key(), &key),
    //         Occupied(_) => panic!(),
    //     }

    //     assert!(!a.contains_key(key));
    //     assert_eq!(a.len(), 0);
    // }

    // #[test]
    // fn test_entry_and_replace_entry_with() {
    //     let mut a = HsHashMap::new();

    //     let key = "a key";
    //     let value = "an initial value";
    //     let new_value = "a new value";

    //     let entry = a.entry(key).and_replace_entry_with(|_, _| panic!());

    //     match entry {
    //         Vacant(e) => assert_eq!(e.key(), &key),
    //         Occupied(_) => panic!(),
    //     }

    //     a.insert(key, value);

    //     let entry = a.entry(key).and_replace_entry_with(|k, v| {
    //         assert_eq!(k, &key);
    //         assert_eq!(v, value);
    //         Some(new_value)
    //     });

    //     match entry {
    //         Occupied(e) => {
    //             assert_eq!(e.key(), &key);
    //             assert_eq!(e.get(), &new_value);
    //         }
    //         Vacant(_) => panic!(),
    //     }

    //     assert_eq!(a[key], new_value);
    //     assert_eq!(a.len(), 1);

    //     let entry = a.entry(key).and_replace_entry_with(|k, v| {
    //         assert_eq!(k, &key);
    //         assert_eq!(v, new_value);
    //         None
    //     });

    //     match entry {
    //         Vacant(e) => assert_eq!(e.key(), &key),
    //         Occupied(_) => panic!(),
    //     }

    //     assert!(!a.contains_key(key));
    //     assert_eq!(a.len(), 0);
    // }

    // #[test]
    // fn test_replace_entry_with_doesnt_corrupt() {
    //     #![allow(deprecated)] //rand
    //                           // Test for #19292
    //     fn check(m: &HsHashMap<i32, ()>) {
    //         for k in m.keys() {
    //             assert!(m.contains_key(k), "{} is in keys() but not in the map?", k);
    //         }
    //     }

    //     let mut m = HsHashMap::new();

    //     let mut rng = {
    //         let seed = u64::from_le_bytes(*b"testseed");
    //         SmallRng::seed_from_u64(seed)
    //     };

    //     // Populate the map with some items.
    //     for _ in 0..50 {
    //         let x = rng.gen_range(-10..10);
    //         m.insert(x, ());
    //     }

    //     for _ in 0..1000 {
    //         let x = rng.gen_range(-10..10);
    //         m.entry(x).and_replace_entry_with(|_, _| None);
    //         check(&m);
    //     }
    // }

    // #[test]
    // fn test_retain() {
    //     let mut map: HsHashMap<i32, i32> = (0..100).map(|x| (x, x * 10)).collect();

    //     map.retain(|&k, _| k % 2 == 0);
    //     assert_eq!(map.len(), 50);
    //     assert_eq!(map[&2], 20);
    //     assert_eq!(map[&4], 40);
    //     assert_eq!(map[&6], 60);
    // }

    // #[test]
    // fn test_const_with_hasher() {
    //     use core::hash::BuildHasher;
    //     use std::collections::hash_map::DefaultHasher;

    //     #[derive(Clone)]
    //     struct MyHasher;
    //     impl BuildHasher for MyHasher {
    //         type Hasher = DefaultHasher;

    //         fn build_hasher(&self) -> DefaultHasher {
    //             DefaultHasher::new()
    //         }
    //     }

    //     const EMPTY_MAP: HsHashMap<u32, std::string::String, MyHasher> =
    //         HsHashMap::with_hasher(MyHasher);

    //     let mut map = EMPTY_MAP;
    //     map.insert(17, "seventeen".to_owned());
    //     assert_eq!("seventeen", map[&17]);
    // }
}
