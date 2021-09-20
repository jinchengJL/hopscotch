#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_example() {
        assert_eq!(2 + 2, 4);
    }

    const MAP_SIZE: usize = 100000;

    #[bench]
    fn bench_insertion(b: &mut test::Bencher) {
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
    fn bench_find(b: &mut test::Bencher) {
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
    fn bench_clone(b: &mut test::Bencher) {
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
    fn bench_removal(b: &mut test::Bencher) {
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
}
