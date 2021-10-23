use criterion::{criterion_group, criterion_main, Criterion};
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

criterion_group!(
    benches,
    hopscotch_insertion,
    hopscotch_get,
);
criterion_main!(benches);
