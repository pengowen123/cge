init:
	git config core.hooksPath .githooks

test:
	cargo test
	cargo build --no-default-features
	cargo build --no-default-features --features "serde"
	cargo build --no-default-features --features "json"

run-examples:
	for eg in `ls ./examples/*.rs | xargs basename --suffix=.rs`; do \
		cargo run --example $$eg; \
	done
