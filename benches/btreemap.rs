use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::BTreeMap;

const MAP_SIZE: usize = 100000;

fn btreemap_insertion(c: &mut Criterion) {
    c.bench_function("btreemap insertion", |b| {
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
    });
}

fn btreemap_get(c: &mut Criterion) {
    let mut map: BTreeMap<i32, i32> = BTreeMap::new();
    for key in 0..MAP_SIZE as i32 {
        map.insert(key * 2, key * 2);
    }
    c.bench_function("btreemap get", |b| {
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

criterion_group!(
    benches,
    btreemap_insertion,
    btreemap_get
);
criterion_main!(benches);
