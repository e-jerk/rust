fn main() {
    let mut v = Vec::with_capacity(4);
    v.push(1u8);
    v.push(2);
    assert_eq!(v.len(), 2);
    println!("ok");
}
