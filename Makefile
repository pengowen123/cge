init:
	git config core.hooksPath .githooks

run-examples:
	for eg in `ls ./examples/*.rs | xargs basename --suffix=.rs`; do \
		cargo run --example $$eg; \
	done
