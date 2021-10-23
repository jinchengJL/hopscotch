use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use hopscotch::HsHashMap;

const MAP_SIZE: usize = 100000;

fn hopscotch_insertion(c: &mut Criterion) {
    c.bench_function("hopscotch insertion", |b| {
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
    });
}

fn hopscotch_get(c: &mut Criterion) {
    let mut map: HsHashMap<i32, i32> = HsHashMap::new();
    for key in 0..MAP_SIZE as i32 {
        map.insert(key * 2, key * 2);
    }
    c.bench_function("hopscotch get", |b| {
        let mut key: i32 = 0;
        b.iter(|| {
            map.get(&key);
            key += 1;
            if key as usize == MAP_SIZE * 2 {
                key = 0;
            }
        })
    });
}

fn hopscotch_remove1000(c: &mut Criterion) {
    let mut map: HsHashMap<i32, i32> = HsHashMap::new();
    for key in 0..MAP_SIZE as i32 {
        map.insert(key, key);
    }
    // Unfortunately, I haven't found a way to tell Criterion to use the the 
    // same input for X iterations before re-generating it. Maybe, use a Boxed
    // hash map and re-generate it in setup only when it's empty?
    c.bench_function("hopscotch remove1000", |b| {
        b.iter_batched_ref(
            || map.clone(),
            |map| {
                for key in 0..1000 as i32 {
                    map.remove(&key);
                }
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(
    benches,
    hopscotch_insertion,
    hopscotch_get,
    hopscotch_remove1000,
);
criterion_main!(benches);
