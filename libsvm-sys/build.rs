extern crate gcc;

fn main() {
    gcc::Config::new()
        .cpp(true)
        .file("libsvm/svm.cpp")
        .compile("libsvm.a");
}
