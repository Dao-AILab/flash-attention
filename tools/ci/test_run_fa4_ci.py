import run_fa4_ci


def test_default_targets_include_split_smoke_suites():
    args = run_fa4_ci.make_parser().parse_args([])
    steps = run_fa4_ci.build_step_plan(
        test_target=args.test_target,
        test_filter=args.test_filter,
        compile_workers=args.compile_workers,
        run_workers=args.run_workers,
        test_visible_devices="0",
        benchmark_visible_devices="0",
        skip_benchmark=True,
    )

    assert steps[0].command[3:6] == [
        "tests/cute/test_flash_attn.py",
        "tests/cute/test_flash_attn_mla.py",
        "tests/cute/test_cuda_graph.py",
    ]


def test_test_target_accepts_one_or_multiple_paths():
    parser = run_fa4_ci.make_parser()

    assert parser.parse_args(["--test-target", "one.py"]).test_target == ["one.py"]
    assert parser.parse_args(["--test-target", "one.py", "two.py"]).test_target == [
        "one.py",
        "two.py",
    ]


def test_step_plan_applies_targets_filter_and_durations_to_both_passes():
    steps = run_fa4_ci.build_step_plan(
        test_target=["one.py", "two.py"],
        test_filter="selected_case or another_case",
        compile_workers=8,
        run_workers=1,
        test_visible_devices="0",
        benchmark_visible_devices="0",
        skip_benchmark=True,
    )

    compile_pass, run_pass = steps
    for step in (compile_pass, run_pass):
        assert step.command[3:5] == ["one.py", "two.py"]
        assert step.command[5:7] == ["-k", "selected_case or another_case"]
        assert "--durations=20" in step.command

    assert compile_pass.command[-3:] == ["-n", "8", "-x"]
    assert compile_pass.extra_env == {
        "FLASH_ATTENTION_FAKE_TENSOR": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert run_pass.extra_env == {
        "FLASH_ATTENTION_FAKE_TENSOR": "0",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def test_step_plan_keeps_single_string_target_compatible():
    steps = run_fa4_ci.build_step_plan(
        test_target="one.py",
        test_filter="",
        compile_workers=1,
        run_workers=1,
        test_visible_devices="0",
        benchmark_visible_devices="0",
        skip_benchmark=True,
    )

    assert steps[0].command[3] == "one.py"
