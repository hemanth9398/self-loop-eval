"""Real-world coding problems for the self-eval loop.

These are practical, interview-level problems that test algorithmic thinking,
data structures, edge case handling, and code quality — not toy examples.
"""

from self_loop_eval.environments.coding import CodingTask, TestCase


def get_real_world_problems() -> list[CodingTask]:
    """Return a curated set of real-world coding problems."""
    return [
        # -------------------------------------------------------------------
        # 1. LRU Cache — real system design pattern
        # -------------------------------------------------------------------
        CodingTask(
            task_id="lru_cache",
            description=(
                "Implement an LRU (Least Recently Used) Cache class.\n\n"
                "The cache has a fixed capacity. When adding a new key beyond capacity,\n"
                "evict the least recently used item.\n\n"
                "Implement a function `lru_cache_ops(capacity, operations)` where:\n"
                "- `capacity` is the max cache size\n"
                "- `operations` is a list of tuples:\n"
                "  - ('put', key, value) — insert or update\n"
                "  - ('get', key) — return value or -1 if not found\n"
                "- Return a list of results for 'get' operations only.\n\n"
                "Both get and put must run in O(1) average time."
            ),
            function_name="lru_cache_ops",
            test_cases=[
                TestCase(
                    input=(
                        "2, [('put', 1, 1), ('put', 2, 2), ('get', 1), "
                        "('put', 3, 3), ('get', 2), ('get', 3)]"
                    ),
                    expected_output="[1, -1, 3]",
                    description="Basic LRU eviction",
                ),
                TestCase(
                    input=(
                        "1, [('put', 1, 10), ('get', 1), ('put', 2, 20), ('get', 1), ('get', 2)]"
                    ),
                    expected_output="[10, -1, 20]",
                    description="Capacity 1 — immediate eviction",
                ),
                TestCase(
                    input=(
                        "2, [('put', 1, 1), ('put', 2, 2), ('get', 1), "
                        "('put', 2, 20), ('get', 2)]"
                    ),
                    expected_output="[1, 20]",
                    description="Update existing key",
                ),
            ],
            ground_truth=(
                "def lru_cache_ops(capacity, operations):\n"
                "    from collections import OrderedDict\n"
                "    cache = OrderedDict()\n"
                "    results = []\n"
                "    for op in operations:\n"
                "        if op[0] == 'get':\n"
                "            key = op[1]\n"
                "            if key in cache:\n"
                "                cache.move_to_end(key)\n"
                "                results.append(cache[key])\n"
                "            else:\n"
                "                results.append(-1)\n"
                "        elif op[0] == 'put':\n"
                "            key, value = op[1], op[2]\n"
                "            if key in cache:\n"
                "                cache.move_to_end(key)\n"
                "            cache[key] = value\n"
                "            if len(cache) > capacity:\n"
                "                cache.popitem(last=False)\n"
                "    return results\n"
            ),
        ),
        # -------------------------------------------------------------------
        # 2. Rate Limiter — real production pattern
        # -------------------------------------------------------------------
        CodingTask(
            task_id="rate_limiter",
            description=(
                "Implement a sliding window rate limiter.\n\n"
                "Write `rate_limiter(max_requests, window_seconds, request_times)` where:\n"
                "- `max_requests`: maximum allowed requests in the window\n"
                "- `window_seconds`: size of the sliding window\n"
                "- `request_times`: list of float timestamps (in seconds) of incoming requests\n"
                "- Return a list of booleans: True if the request is allowed, "
                "False if rate-limited.\n\n"
                "A request at time T is allowed if fewer than `max_requests` occurred in\n"
                "(T - window_seconds, T]."
            ),
            function_name="rate_limiter",
            test_cases=[
                TestCase(
                    input="3, 10, [1, 2, 3, 4, 11, 12]",
                    expected_output="[True, True, True, False, True, True]",
                    description="Basic rate limiting",
                ),
                TestCase(
                    input="1, 5, [1, 2, 6, 7]",
                    expected_output="[True, False, True, False]",
                    description="Single request window",
                ),
                TestCase(
                    input="2, 1, [0.0, 0.5, 1.0, 1.5, 2.0]",
                    expected_output="[True, True, True, True, True]",
                    description="Fast window expiry",
                ),
            ],
            ground_truth=(
                "def rate_limiter(max_requests, window_seconds, request_times):\n"
                "    from collections import deque\n"
                "    window = deque()\n"
                "    results = []\n"
                "    for t in request_times:\n"
                "        while window and window[0] <= t - window_seconds:\n"
                "            window.popleft()\n"
                "        if len(window) < max_requests:\n"
                "            window.append(t)\n"
                "            results.append(True)\n"
                "        else:\n"
                "            results.append(False)\n"
                "    return results\n"
            ),
        ),
        # -------------------------------------------------------------------
        # 3. Merge Intervals — classic real-world scheduling problem
        # -------------------------------------------------------------------
        CodingTask(
            task_id="merge_intervals",
            description=(
                "Given a list of intervals `[start, end]`, merge all overlapping intervals\n"
                "and return the list of merged intervals sorted by start time.\n\n"
                "Write `merge_intervals(intervals)` that returns the merged list.\n\n"
                "Example: [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]"
            ),
            function_name="merge_intervals",
            test_cases=[
                TestCase(
                    input="[[1,3],[2,6],[8,10],[15,18]]",
                    expected_output="[[1, 6], [8, 10], [15, 18]]",
                    description="Standard merge",
                ),
                TestCase(
                    input="[[1,4],[4,5]]",
                    expected_output="[[1, 5]]",
                    description="Touching intervals",
                ),
                TestCase(
                    input="[[1,4],[0,4]]",
                    expected_output="[[0, 4]]",
                    description="Unsorted input",
                ),
                TestCase(
                    input="[]",
                    expected_output="[]",
                    description="Empty input",
                ),
                TestCase(
                    input="[[1,4],[2,3]]",
                    expected_output="[[1, 4]]",
                    description="Contained interval",
                ),
            ],
            ground_truth=(
                "def merge_intervals(intervals):\n"
                "    if not intervals:\n"
                "        return []\n"
                "    intervals.sort(key=lambda x: x[0])\n"
                "    merged = [intervals[0]]\n"
                "    for start, end in intervals[1:]:\n"
                "        if start <= merged[-1][1]:\n"
                "            merged[-1][1] = max(merged[-1][1], end)\n"
                "        else:\n"
                "            merged.append([start, end])\n"
                "    return merged\n"
            ),
        ),
        # -------------------------------------------------------------------
        # 4. Serialize/Deserialize Binary Tree — real systems problem
        # -------------------------------------------------------------------
        CodingTask(
            task_id="flatten_nested_dict",
            description=(
                "Write `flatten_dict(d, sep='.')` that flattens a nested dictionary.\n\n"
                "Nested keys are joined with `sep`. Lists should use index notation.\n\n"
                "Example:\n"
                "  {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}} →\n"
                "  {'a': 1, 'b.c': 2, 'b.d.e': 3}\n\n"
                "  {'x': [1, 2], 'y': {'z': [3]}} →\n"
                "  {'x.0': 1, 'x.1': 2, 'y.z.0': 3}"
            ),
            function_name="flatten_dict",
            test_cases=[
                TestCase(
                    input="{'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}",
                    expected_output="{'a': 1, 'b.c': 2, 'b.d.e': 3}",
                    description="Basic nesting",
                ),
                TestCase(
                    input="{'x': [1, 2], 'y': {'z': [3]}}",
                    expected_output="{'x.0': 1, 'x.1': 2, 'y.z.0': 3}",
                    description="Lists with index notation",
                ),
                TestCase(
                    input="{}",
                    expected_output="{}",
                    description="Empty dict",
                ),
                TestCase(
                    input="{'a': {'b': {'c': {'d': 1}}}}",
                    expected_output="{'a.b.c.d': 1}",
                    description="Deep nesting",
                ),
            ],
            ground_truth=(
                "def flatten_dict(d, sep='.', prefix=''):\n"
                "    items = {}\n"
                "    if isinstance(d, dict):\n"
                "        for k, v in d.items():\n"
                "            new_key = f'{prefix}{sep}{k}' if prefix else k\n"
                "            items.update(flatten_dict(v, sep, new_key))\n"
                "    elif isinstance(d, list):\n"
                "        for i, v in enumerate(d):\n"
                "            new_key = f'{prefix}{sep}{i}' if prefix else str(i)\n"
                "            items.update(flatten_dict(v, sep, new_key))\n"
                "    else:\n"
                "        items[prefix] = d\n"
                "    return items\n"
            ),
        ),
        # -------------------------------------------------------------------
        # 5. Topological Sort — real dependency resolution
        # -------------------------------------------------------------------
        CodingTask(
            task_id="task_scheduler",
            description=(
                "You have `n` tasks labeled 0..n-1 and a list of dependency pairs\n"
                "`[a, b]` meaning task `a` must complete before task `b`.\n\n"
                "Write `task_order(n, dependencies)` that returns a valid execution order\n"
                "(list of task labels), or an empty list if there's a circular dependency.\n\n"
                "If multiple valid orders exist, return any one."
            ),
            function_name="task_order",
            test_cases=[
                TestCase(
                    input="4, [[1,0],[2,0],[3,1],[3,2]]",
                    expected_output="[3, 1, 2, 0]",
                    description="Diamond dependency",
                ),
                TestCase(
                    input="2, [[1,0],[0,1]]",
                    expected_output="[]",
                    description="Circular dependency",
                ),
                TestCase(
                    input="3, []",
                    expected_output="[0, 1, 2]",
                    description="No dependencies",
                ),
            ],
            ground_truth=(
                "def task_order(n, dependencies):\n"
                "    from collections import deque\n"
                "    graph = [[] for _ in range(n)]\n"
                "    in_degree = [0] * n\n"
                "    for a, b in dependencies:\n"
                "        graph[a].append(b)\n"
                "        in_degree[b] += 1\n"
                "    queue = deque(i for i in range(n) if in_degree[i] == 0)\n"
                "    order = []\n"
                "    while queue:\n"
                "        node = queue.popleft()\n"
                "        order.append(node)\n"
                "        for neighbor in graph[node]:\n"
                "            in_degree[neighbor] -= 1\n"
                "            if in_degree[neighbor] == 0:\n"
                "                queue.append(neighbor)\n"
                "    return order if len(order) == n else []\n"
            ),
        ),
    ]
