"""Behavioural tests. These use the real all-MiniLM-L6-v2 model (downloaded once
and cached) so the semantic ranking is genuinely exercised, not mocked."""

SAMPLE_DOCS = [
    {
        "title": "RCS Thruster Firing Procedure",
        "content": "This procedure describes the cold gas thruster ignition sequence "
        "for the reaction control system. It covers pre-ignition checks, valve "
        "actuation order, and post-firing telemetry verification.",
    },
    {
        "title": "Solar Panel Deployment Sequence",
        "content": "Detailed steps for deploying the photovoltaic arrays after orbital "
        "insertion. Includes hinge release commands, deployment angle monitoring, "
        "and power generation confirmation checks.",
    },
    {
        "title": "Thermal Control System Overview",
        "content": "Description of the passive and active thermal regulation mechanisms "
        "onboard Nyx. Covers heat pipe routing, multi-layer insulation zones, and "
        "heater activation thresholds.",
    },
    {
        "title": "Communication Link Budget",
        "content": "Analysis of the S-band uplink and downlink margins. Includes antenna "
        "gain figures, free-space path loss calculations, and minimum required Eb/N0 "
        "for command reception.",
    },
    {
        "title": "Propellant Tank Pressurisation Procedure",
        "content": "Step-by-step guide for pressurising the hydrazine tank prior to "
        "orbital manoeuvre. Covers regulator settings, pressure transducer readings, "
        "and abort criteria.",
    },
]


def seed(client):
    for doc in SAMPLE_DOCS:
        assert client.post("/documents", json=doc).status_code == 201


def test_create_returns_id_and_timestamp(client):
    response = client.post("/documents", json=SAMPLE_DOCS[0])
    assert response.status_code == 201
    body = response.json()
    assert body["id"] == 1
    assert body["title"] == SAMPLE_DOCS[0]["title"]
    assert body["created_at"]  # set automatically on creation
    assert "embedding" not in body  # internal field stays internal


def test_semantic_search_ranks_related_above_unrelated(client):
    seed(client)
    # "rocket propulsion" shares no words with "cold gas thruster / reaction
    # control", yet must rank that doc above the comms and solar-panel docs.
    results = client.get(
        "/documents/search", params={"q": "rocket propulsion"}
    ).json()
    titles = [r["title"] for r in results]
    assert titles.index("RCS Thruster Firing Procedure") < titles.index(
        "Communication Link Budget"
    )
    assert titles.index("RCS Thruster Firing Procedure") < titles.index(
        "Solar Panel Deployment Sequence"
    )
    # Results are returned in descending score order.
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results(client):
    seed(client)
    results = client.get(
        "/documents/search", params={"q": "spacecraft", "top_k": 2}
    ).json()
    assert len(results) == 2


def test_delete_then_404(client):
    created = client.post("/documents", json=SAMPLE_DOCS[0]).json()
    assert client.delete(f"/documents/{created['id']}").status_code == 204
    assert client.delete(f"/documents/{created['id']}").status_code == 404


def test_filter_title_restricts_candidates(client):
    seed(client)
    results = client.get(
        "/documents/search", params={"q": "procedure", "filter_title": "thruster"}
    ).json()
    assert len(results) == 1
    assert results[0]["title"] == "RCS Thruster Firing Procedure"


def test_invalid_inputs_return_422(client):
    assert client.get("/documents/search").status_code == 422  # missing q
    assert (
        client.post("/documents", json={"title": "", "content": ""}).status_code == 422
    )


def test_rank_one_correctness(client):
    """Stronger than the relative-order test: each query's *top* hit must be the
    semantically correct doc, across all five topics."""
    seed(client)
    expectations = {
        "deploying the solar arrays": "Solar Panel Deployment Sequence",
        "pressurise the fuel tank": "Propellant Tank Pressurisation Procedure",
        "radio antenna signal strength": "Communication Link Budget",
        "temperature regulation": "Thermal Control System Overview",
        "how do I fire the thrusters": "RCS Thruster Firing Procedure",
    }
    for query, expected in expectations.items():
        results = client.get("/documents/search", params={"q": query}).json()
        assert results[0]["title"] == expected, (query, results[0]["title"])


def test_blank_inputs_return_422(client):
    # whitespace-only is no longer silently accepted (stripped -> empty -> 422)
    assert client.post("/documents", json={"title": "   ", "content": "x"}).status_code == 422
    assert client.post("/documents", json={"title": "x", "content": "   "}).status_code == 422
    assert client.get("/documents/search", params={"q": "   "}).status_code == 422
    assert (
        client.get("/documents/search", params={"q": "x", "filter_title": "   "}).status_code
        == 422
    )


def test_duplicate_document_returns_409(client):
    assert client.post("/documents", json=SAMPLE_DOCS[0]).status_code == 201
    # exact duplicate
    assert client.post("/documents", json=SAMPLE_DOCS[0]).status_code == 409
    # surrounding whitespace is stripped first, so this is also a duplicate
    padded = {
        "title": f"  {SAMPLE_DOCS[0]['title']}  ",
        "content": SAMPLE_DOCS[0]["content"],
    }
    assert client.post("/documents", json=padded).status_code == 409
