N, M = map(int, input().split(" "))
adj = [[] for _ in range(N+1)]
edges = []
for eid in range(M-1):
    a,b = map(int, input().split(" "))
    edges.append((a,b))
    adj[a].append((b,eid))
    adj[b].append((a,eid))

tin = [0]*(N+1)
used_edge = [False]*M
time = 0
edge_stack = []
cycles = []
visited = [False]*(N+1)

def dfs_find(u, peid):
    global time
    time += 1
    tin[u] = time
    visited[u] = True
    for v,eid in adj[u]:
        if eid == peid:
            continue
        if not used_edge[eid]:
            edge_stack.append((eid, u, v))
            used_edge[eid] = True
        if not visited[v]:
            dfs_find(v, eid)
        else:
            if tin[v] < tin[u]:
                collected_edges = []
                while edge_stack:
                    e,eu,ev = edge_stack.pop()
                    collected_edges.append((e, eu, ev))
                    if (eu == u and ev == v) or (eu == v and ev == u) or (ev == v and eu == u) or (ev == u and eu == v):
                        break
                verts_set = set()
                for e,eu,ev in collected_edges:
                    verts_set.add(eu); verts_set.add(ev)
                small_adj = {}
                for e,eu,ev in collected_edges:
                    small_adj.setdefault(eu, []).append(ev)
                    small_adj.setdefault(ev, []).append(eu)
                start = next(iter(verts_set))
                order = []
                prev = -1
                cur = start
                while True:
                    order.append(cur)
                    neighs = small_adj[cur]
                    nxt = neighs[0] if neighs[0] != prev else (neighs[1] if len(neighs) > 1 else None)
                    if nxt is None:
                        break
                    prev, cur = cur, nxt
                    if cur == start:
                        break
                cycles.append(order)

for i in range(1, N+1):
    if not visited[i]:
        dfs_find(i, -1)

in_cycle = [False]*(N+1)
cycle_of_vertex = [-1]*(N+1)
cycle_vertices = []
for cid, cyc in enumerate(cycles):
    for v in cyc:
        in_cycle[v] = True
        cycle_of_vertex[v] = cid
    cycle_vertices.append(cyc)

comp_id = [-1]*(N+1)
comp_cnt = 0
for cid, cyc in enumerate(cycle_vertices):
    compid = comp_cnt
    comp_cnt += 1
    for v in cyc:
        comp_id[v] = compid
for v in range(1, N+1):
    if comp_id[v] == -1:
        comp_id[v] = comp_cnt
        comp_cnt += 1

comp_adj = [[] for _ in range(comp_cnt)]
for (u,v) in edges:
    cu, cv = comp_id[u], comp_id[v]
    if cu != cv:
        comp_adj[cu].append((cv, u, v))
        comp_adj[cv].append((cu, v, u))

comp_is_cycle = [False]*comp_cnt
comp_cycle_list = [None]*comp_cnt
comp_vertex_index = [None]*comp_cnt
for cid, cyc in enumerate(cycle_vertices):
    comp = cid
    comp_is_cycle[comp] = True
    comp_cycle_list[comp] = cyc[:]
    idxmap = {}
    for i,v in enumerate(cyc):
        idxmap[v] = i
    comp_vertex_index[comp] = idxmap

comp_single_vertex = [None]*comp_cnt
for v in range(1, N+1):
    c = comp_id[v]
    if not comp_is_cycle[c]:
        comp_single_vertex[c] = v

seen_comp = [False]*comp_cnt

def dfs_comp(comp, parent, attach_vertex):
    seen_comp[comp] = True
    if not comp_is_cycle[comp]:
        v = comp_single_vertex[comp]
        total = 0
        for (nbr, self_v, nbr_v) in comp_adj[comp]:
            if nbr == parent: continue
            child_val = dfs_comp(nbr, comp, nbr_v)
            total += child_val + 1
        return total
    else:
        cyc = comp_cycle_list[comp]
        k = len(cyc)
        w = [0]*k
        idxmap = comp_vertex_index[comp]
        for (nbr, self_v, nbr_v) in comp_adj[comp]:
            if nbr == parent: continue
            i = idxmap[self_v]
            child_val = dfs_comp(nbr, comp, nbr_v)
            w[i] += child_val + 1
        B = [w[i] - i for i in range(k)]
        C = [w[i] + i for i in range(k)]
        pre_best = [(-10**18,-1, -10**18,-1)] * k
        best0 = -10**18; id0 = -1; best1 = -10**18; id1 = -1
        for i in range(k):
            val = B[i]
            if val > best0:
                best1, id1 = best0, id0
                best0, id0 = val, i
            elif val > best1:
                best1, id1 = val, i
            pre_best[i] = (best0, id0, best1, id1)
        suf_best = [(-10**18,-1, -10**18,-1)] * (k+1)
        best0 = -10**18; id0 = -1; best1 = -10**18; id1 = -1
        for i in range(k-1, -1, -1):
            val = B[i]
            if val > best0:
                best1, id1 = best0, id0
                best0, id0 = val, i
            elif val > best1:
                best1, id1 = val, i
            suf_best[i] = (best0, id0, best1, id1)
        preC = [(-10**18,-1, -10**18,-1)] * k
        best0 = -10**18; id0 = -1; best1 = -10**18; id1 = -1
        for i in range(k):
            val = C[i]
            if val > best0:
                best1, id1 = best0, id0
                best0, id0 = val, i
            elif val > best1:
                best1, id1 = val, i
            preC[i] = (best0, id0, best1, id1)
        sufC = [(-10**18,-1, -10**18,-1)] * (k+1)
        best0 = -10**18; id0 = -1; best1 = -10**18; id1 = -1
        for i in range(k-1, -1, -1):
            val = C[i]
            if val > best0:
                best1, id1 = best0, id0
                best0, id0 = val, i
            elif val > best1:
                best1, id1 = val, i
            sufC[i] = (best0, id0, best1, id1)
        def T1_excl(p):
            a_val, a_idx, a_val2, a_idx2 = pre_best[p]
            candA_val = -10**18; candA_idx = -1
            if a_idx != p:
                candA_val, candA_idx = a_val, a_idx
            elif a_idx2 != -1:
                candA_val, candA_idx = a_val2, a_idx2
            if p+1 <= k-1:
                b_val, b_idx, b_val2, b_idx2 = suf_best[p+1]
                candB_val = b_val; candB_idx = b_idx
            else:
                candB_val = -10**18; candB_idx = -1
            best_val = -10**18; best_idx = -1
            if candA_idx != -1:
                vv = p + candA_val
                best_val = vv; best_idx = candA_idx
            if candB_idx != -1:
                vv = p + k + candB_val
                if vv > best_val:
                    best_val = vv; best_idx = candB_idx
            return best_val, best_idx
        def T2_excl(p):
            a_val, a_idx, a_val2, a_idx2 = sufC[p]
            candA_val = a_val; candA_idx = a_idx
            candA_val2 = a_val2; candA_idx2 = a_idx2
            if candA_idx == p:
                if candA_idx2 != -1:
                    candA_val = candA_val2; candA_idx = candA_idx2
                else:
                    candA_idx = -1
            if p-1 >= 0:
                b_val, b_idx, b_val2, b_idx2 = preC[p-1]
                candB_val = b_val; candB_idx = b_idx
            else:
                candB_val = -10**18; candB_idx = -1
            best_val = -10**18; best_idx = -1
            if candA_idx != -1:
                vv = -p + candA_val
                best_val = vv; best_idx = candA_idx
            if candB_idx != -1:
                vv = -p + k + candB_val
                if vv > best_val:
                    best_val = vv; best_idx = candB_idx
            return best_val, best_idx
        p_idx = comp_vertex_index[comp][attach_vertex] if attach_vertex is not None else 0
        t1_val, t1_idx = T1_excl(p_idx)
        t2_val, t2_idx = T2_excl(p_idx)
        M_excl = max(t1_val, t2_val, -10**18)
        if M_excl < -10**17:
            M_excl = -10**18
        base_no_cycle = w[p_idx]
        base_full_cycle = w[p_idx] + k
        candidate = base_no_cycle
        candidate = max(candidate, base_full_cycle)
        if M_excl > -10**17:
            candidate = max(candidate, w[p_idx] + M_excl)
        return candidate

root_comp = comp_id[1]
ans = dfs_comp(root_comp, -1, 1)
print(ans)
