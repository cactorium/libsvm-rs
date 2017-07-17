extern crate gcc;

fn main() {
    gcc::Config::new()
        .cpp(true)
        .file("libsvm/libsvm.cpp")
        .compile("libsvm.a");
}
