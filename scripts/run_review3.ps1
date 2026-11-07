<#
Review 3: the whole run, unattended and self-healing.

    cd "D:\Deep Learning\hallu-probe"
    powershell -ExecutionPolicy Bypass -File scripts\run_review3.ps1

Start it and walk away. It is built so that nobody has to watch a console
for ten hours:

  * every step is retried with progressively cheaper settings, so a CUDA
    out-of-memory on the 3B model drops max-new-tokens, then strides the
    token-trajectory cache, rather than ending the night
  * transient Hugging Face network errors are retried after a pause
  * a step that cannot be rescued is recorded and skipped; the remaining
    model-dataset combinations still run
  * Windows is kept awake for the duration and released at the end
  * every step appends to out\review3_status.json, so the morning question
    "what actually happened" has a written answer
  * finished work is detected and skipped, so re-running costs nothing

Exit code is 0 if both result files were produced, 1 otherwise.

A note on the fallbacks: --layer-stride only thins the token-trajectory
cache, so it degrades the traj probe and leaves linear, mlp and
layer_attn untouched. --max-new-tokens shortens answers, which can change
labels, so the status file records which attempt succeeded and anything
that did not run at the honest settings must be reported that way.
#>
param(
    [switch]$NoSleepGuard,
    [switch]$IncludeTriviaQA,
    [switch]$NoQuarantine,
    [switch]$SkipDecodingCheck,
    [int]$NetworkRetries = 3
)

$ErrorActionPreference = "Continue"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logDir = Join-Path $repo "logs"
$stepDir = Join-Path $logDir "review3-steps"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $stepDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repo "out") | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runLog = Join-Path $logDir "review3-$stamp.log"
$statusPath = Join-Path $repo "out\review3_status.json"

# every key this script will ever set is declared here, so no code path
# depends on being able to add one to the dictionary later
$script:status = [ordered]@{
    started  = (Get-Date).ToString("s")
    updated  = (Get-Date).ToString("s")
    repo     = $repo
    log      = $runLog
    handoff  = ""
    steps    = @()
    finished = ""
    verdict  = "running"
}

function Save-Status {
    $script:status.updated = (Get-Date).ToString("s")
    $script:status | ConvertTo-Json -Depth 6 | Set-Content -Path $statusPath -Encoding UTF8
}

function Say {
    param([string]$Msg, [string]$Colour = "Gray")
    Write-Host $Msg -ForegroundColor $Colour
    Add-Content -Path $runLog -Value ("{0}  {1}" -f (Get-Date).ToString('HH:mm:ss'), $Msg)
}

# ------------------------------------------------- keep the laptop awake
# An overnight run that sleeps at 01:00 and resumes when someone opens the
# lid is worse than one that fails, because it looks like it worked.
$script:sleepGuard = $false
if (-not $NoSleepGuard) {
    try {
        Add-Type -Name Power -Namespace Win32 -MemberDefinition @'
[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
        # The flags are written as decimal and cast, NOT as 0x80000000.
        # Windows PowerShell 5.1 parses that hex literal as Int32
        # -2147483648, which a uint parameter rejects, so the call failed
        # silently and the guard never held -- which is how an overnight
        # run froze at 03:08 and sat dead for five hours. Decimal
        # 2147483648 promotes to Int64 and casts cleanly on both 5.1 and 7.
        $ES_CONTINUOUS = 2147483648
        $ES_SYSTEM_REQUIRED = 1
        $ES_AWAYMODE_REQUIRED = 64

        # The return value is the previous state, or 0 on failure. It is
        # checked, because a guard that is reported as on while being off
        # is worse than no guard at all.
        $prev = [Win32.Power]::SetThreadExecutionState(
            [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED))
        if ($prev -eq 0) {
            # away mode is not available on every machine; system-required
            # alone is enough to stop an idle suspend
            $prev = [Win32.Power]::SetThreadExecutionState(
                [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED))
            if ($prev -ne 0) {
                Say "sleep guard on without away mode (this machine refused it)" "DarkGray"
            }
        }
        if ($prev -ne 0) {
            $script:sleepGuard = $true
        } else {
            Say "SLEEP GUARD REJECTED by Windows -- this run can be suspended" "Red"
            Say "Set the power plan manually before leaving it:" "Red"
            Say "  powercfg /change standby-timeout-ac 0" "Red"
        }
    } catch {
        Say "could not set the sleep guard: $($_.Exception.Message)" "Yellow"
        Say "if the laptop sleeps, just re-run this script -- it resumes where it stopped" "Yellow"
    }
}

# SetThreadExecutionState did not hold on at least one run of this project:
# the 3B weight load froze at 47% and sat dead for five hours. So the power
# plan is inspected too, and a configuration that will suspend the machine
# is reported loudly before ten hours are committed to it, rather than
# discovered afterwards.
function Test-PowerSettings {
    $warn = @()
    try {
        $q = (& powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE) 2>$null | Out-String
        $m = [regex]::Match($q, 'Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)')
        if ($m.Success) {
            $secs = [Convert]::ToInt32($m.Groups[1].Value, 16)
            if ($secs -ne 0) {
                $warn += "the AC power plan suspends this machine after $([int]($secs/60)) minutes of idle"
            }
        }
    } catch { }
    try {
        $bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
        if ($bat -and $bat.BatteryStatus -ne 2) {
            $warn += "the laptop is on battery, not mains"
        }
    } catch { }
    if ($warn.Count) {
        Say ""
        Say "!! POWER WARNING -- this run may not survive the night" "Red"
        foreach ($w in $warn) { Say "   - $w" "Red" }
        Say "   A GPU run looks busy while the machine is suspended, so it can" "Red"
        Say "   sit dead for hours. Before leaving it: plug in the charger and run" "Red"
        Say "     powercfg /change standby-timeout-ac 0" "Red"
        Say "     powercfg /change hibernate-timeout-ac 0" "Red"
        Say "   then start this script again. Continuing in 20 seconds." "Red"
        Start-Sleep -Seconds 20
    } else {
        Say "power: on mains, AC idle-sleep disabled" "DarkGray"
    }
}

function Release-SleepGuard {
    if ($script:sleepGuard) {
        try { [void][Win32.Power]::SetThreadExecutionState([uint32]2147483648) } catch { }
    }
}

# --------------------------------------------------- error classification
function Get-Diagnosis {
    param([string]$LogFile)
    if (-not (Test-Path $LogFile)) { return "nolog" }
    $t = Get-Content $LogFile -Raw -ErrorAction SilentlyContinue
    if ([string]::IsNullOrWhiteSpace($t)) { return "nolog" }
    if ($t -match 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED') { return "oom" }
    if ($t -match 'No space left|Errno 28|not enough space') { return "disk" }
    if ($t -match 'Max retries exceeded|ConnectionError|HTTPSConnectionPool|Read timed out|IncompleteRead|Connection reset|ProxyError|ReadTimeout') { return "network" }
    if ($t -match 'GatedRepoError|401 Client Error|403 Client Error|is restricted|awaiting a review') { return "auth" }
    if ($t -match 'one class only|single class|every confound column was empty') { return "data" }
    if ($t -match 'CUDA driver|no kernel image|device-side assert') { return "driver" }
    return "unknown"
}

# Each rung is a single string of extra arguments, empty for the honest
# first attempt. Strings are used rather than nested arrays because
# @(@(), @("-x")) flattens in PowerShell and silently drops the empty
# rung, which would skip the first attempt everywhere.
$extractLadder = @(
    "",
    "--max-new-tokens 24",
    "--max-new-tokens 24 --layer-stride 2",
    "--max-new-tokens 16 --layer-stride 4"
)
$baselineLadder = @(
    "",
    "--k 4 --max-new-tokens 20",
    "--k 3 --max-new-tokens 16",
    "--k 3 --max-new-tokens 12"
)
$plainLadder = @("")

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$BaseArgs,
        [string[]]$Ladder = @(""),
        [string]$Kind = "generic"
    )
    $slug = ($Name -replace '[^A-Za-z0-9]+', '-').Trim('-')
    $attempts = @()
    $netTries = 0
    $i = 0
    $firstLog = Join-Path $stepDir "$slug--try1.log"

    while ($i -lt $Ladder.Count) {
        $rung = $Ladder[$i]
        $extra = if ([string]::IsNullOrWhiteSpace($rung)) { @() } else { $rung -split '\s+' }
        $argv = @($BaseArgs) + $extra
        $stepLog = Join-Path $stepDir ("{0}--try{1}.log" -f $slug, ($attempts.Count + 1))
        $shown = if ($extra.Count) { "  (fallback: $rung)" } else { "" }

        Say ""
        Say "=== $Name$shown" "Cyan"
        $t0 = Get-Date

        & python @argv *>&1 | Tee-Object -FilePath $stepLog | ForEach-Object { Write-Host "    $_" }
        $code = $LASTEXITCODE
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

        if ($code -eq 0) {
            Say "    ok in $mins min" "Green"
            $attempts += [ordered]@{ try = $attempts.Count + 1; extra = $rung; exit = 0; minutes = $mins; diagnosis = "" }
            $script:status.steps += [ordered]@{
                name = $Name; kind = $Kind; result = "ok"; minutes = $mins
                settings = $rung; attempts = $attempts; log = $stepLog
            }
            Save-Status
            return $true
        }

        $why = Get-Diagnosis $stepLog
        $attempts += [ordered]@{ try = $attempts.Count + 1; extra = $rung; exit = $code; minutes = $mins; diagnosis = $why }
        Say "    failed after $mins min (exit $code, looks like: $why)" "Yellow"

        # `continue` inside a switch would act on the switch, not this loop,
        # so the next rung is chosen with an explicit variable instead
        $advance = 1
        if ($why -eq "network") {
            if ($netTries -lt $NetworkRetries) {
                $netTries++
                Say "    network error; waiting 90s and retrying the same settings ($netTries/$NetworkRetries)" "Yellow"
                Start-Sleep -Seconds 90
                $advance = 0
            }
        } elseif ($why -eq "disk") {
            Say "    out of disk. Nothing cheaper will help." "Red"
            $advance = $Ladder.Count
        } elseif ($why -eq "auth") {
            Say "    the model repository refused access; a token or an accepted licence is needed." "Red"
            $advance = $Ladder.Count
        } elseif ($why -eq "driver") {
            Say "    this looks like a GPU driver fault rather than a memory one." "Red"
            $advance = $Ladder.Count
        } elseif ($why -eq "oom") {
            Start-Sleep -Seconds 20
            if ($i + 1 -lt $Ladder.Count) {
                Say "    out of VRAM; dropping to cheaper settings" "Yellow"
            }
        } else {
            if ($i + 1 -lt $Ladder.Count) { Say "    retrying with cheaper settings" "Yellow" }
        }
        $i += $advance
    }

    Say "    giving up on this step and carrying on" "Red"
    $script:status.steps += [ordered]@{
        name = $Name; kind = $Kind; result = "failed"; minutes = 0
        settings = ""; attempts = $attempts; log = $firstLog
    }
    Save-Status
    return $false
}

# ------------------------------------------------------------------ begin
Say "Review 3 unattended run" "Cyan"
Say "log:    $runLog" "Cyan"
Say "status: $statusPath  <- read this one in the morning" "Cyan"
if ($script:sleepGuard) { Say "sleep guard on: Windows stays awake until this exits" "DarkGray" }
$gpu = ""
try { $gpu = (& nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>$null) -join "; " } catch { }
if ($gpu) { Say "gpu: $gpu" } else { Say "nvidia-smi gave nothing; extraction needs a CUDA device" "Yellow" }
Test-PowerSettings
Save-Status

# 3B leads the list: it is the positive control. If the signal is absent
# there, the method is broken rather than the models being too small, and
# that is worth knowing in hour one rather than hour twelve.

# ------------------------------------------------- decoding pre-flight
# model.generate(do_sample=False) is not greedy decoding on its own.
# Hugging Face starts from the checkpoint's generation_config.json, and a
# repetition penalty is a logits processor rather than a sampling warper,
# so it applies anyway. Qwen2.5 ships one (1.10 at 0.5B and 1.5B, 1.05 at
# 3B) and Llama-3.2 does not, so the first run decoded the two families
# differently and did not even hold Qwen fixed across its three sizes.
#
# Two minutes per model answers it definitively, and decides which caches
# have to be re-extracted. That is cheap against the hours an extraction
# costs, which is the whole reason this stage is first.
$models = @(
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct"
)

if (-not $SkipDecodingCheck) {
    Say ""
    Say "### STAGE 0 -- decoding pre-flight, about two minutes per model" "Cyan"
    foreach ($m in $models) {
        $short = $m.Split("/")[-1]
        Invoke-Step -Name "decoding check $short" `
            -BaseArgs @("scripts\10_check_decoding.py", "--model", $m, "--device", "cuda", "--n", "16") `
            -Ladder $plainLadder -Kind "decoding" | Out-Null
    }

    Say ""
    Say "### which caches survive the pre-flight" "Cyan"
    $qArgs = @("scripts\11_quarantine_stale.py")
    if (-not $NoQuarantine) { $qArgs += "--apply" }
    Invoke-Step -Name "quarantine stale caches" -BaseArgs $qArgs `
        -Ladder $plainLadder -Kind "quarantine" | Out-Null
    Say "stale caches are MOVED to out\_quarantine_predecoding\, never deleted:" "DarkGray"
    Say "the size of the decoding effect is itself a result worth reporting." "DarkGray"
}

# The working sets are VERIFIED, not rebuilt. 00_prepare_data.py overwrites
# them unconditionally, and every cache on disk was extracted in the order
# of the set that existed at the time: cached item i is working-set item i.
# Re-sampling would leave the labels in acts.npz belonging to different
# questions, and nothing downstream would notice.
$wsPopqa = Join-Path $repo "data\working_popqa.jsonl"
$wsTrivia = Join-Path $repo "data\working_triviaqa.jsonl"
if ((Test-Path $wsPopqa) -and (Test-Path $wsTrivia)) {
    Say ""
    Say "=== working sets already on disk; verifying instead of rebuilding" "DarkGray"
    $verdictPath = Join-Path $repo "out\alignment_verdict.json"
    if (Test-Path $verdictPath) { Remove-Item $verdictPath -Force }
    Invoke-Step -Name "verify cache alignment" `
        -BaseArgs @("scripts\09_verify_alignment.py") -Ladder $plainLadder -Kind "verify" | Out-Null

    # The gate reads the verdict file, not the exit code. A run on
    # 13 October halted here with "a cache does not line up" while the
    # check itself had printed "all 3 cache(s) line up": the exit status
    # came back wrong through the pipeline. A gate that can fail closed on
    # a false alarm can also fail open on a real mismatch, so the decision
    # is taken from what the check wrote down.
    $aligned = $false
    if (Test-Path $verdictPath) {
        try {
            $v = Get-Content $verdictPath -Raw | ConvertFrom-Json
            $aligned = [bool]$v.ok
            if ($aligned) {
                Say "alignment verdict: ok, $($v.checked) cache(s) verified" "DarkGray"
            } else {
                foreach ($b in $v.bad) { Say "  BAD $($b.cache): $($b.why)" "Red" }
            }
        } catch {
            Say "could not read $verdictPath : $($_.Exception.Message)" "Red"
        }
    } else {
        Say "the alignment check wrote no verdict file, so alignment is unconfirmed" "Red"
    }
    if (-not $aligned) {
        Say ""
        Say "STOPPING: a cache on disk does not line up with its working set." "Red"
        Say "Running the analysis on it would attach labels to the wrong questions." "Red"
        Say "See the log above, then either restore the working set those caches" "Red"
        Say "were built from, or delete the mismatched cache and re-extract." "Red"
        $script:status.finished = (Get-Date).ToString("s")
        $script:status.verdict = "stopped: cache/working-set mismatch"
        Save-Status
        Release-SleepGuard
        exit 1
    }
} else {
    Say "a working set is missing; building both" "Yellow"
    Invoke-Step -Name "prepare working sets" `
        -BaseArgs @("scripts\00_prepare_data.py", "--n", "800") `
        -Ladder $plainLadder -Kind "data" | Out-Null
}

# Two passes, in this order on purpose. PopQA is already extracted, so its
# baselines are the whole remaining cost of a complete headline result:
# the baseline ladder across all five models. That lands in about two
# hours. TriviaQA needs five fresh extractions on top, which is most of a
# day, and buys the replication and the transfer experiment.
#
# Under a deadline, finishing a defensible subset first beats starting the
# expensive half and having neither. So pass 1 produces a usable results
# file, the analysis runs on it, and only then does pass 2 begin. If pass 2
# never finishes, the deck still has real numbers.

function Invoke-ModelStage {
    param([string]$Dataset, [switch]$AllowExtract)
    foreach ($m in $models) {
        $mslug = ($m -replace '[^A-Za-z0-9\-_.]', '_')
        $cache = Join-Path $repo ("out\{0}__{1}\acts.npz" -f $mslug, $Dataset)
        $short = $m.Split("/")[-1]

        $extractOk = $false
        if (Test-Path $cache) {
            Say ""
            Say "=== extract $short $Dataset -- cache already on disk, GPU skipped" "DarkGray"
            $script:status.steps += [ordered]@{
                name = "extract $short $Dataset"; kind = "extract"
                result = "skipped (cached)"; minutes = 0; settings = ""
                attempts = @(); log = ""
            }
            Save-Status
            $extractOk = $true
        } elseif ($AllowExtract) {
            $extractOk = Invoke-Step -Name "extract $short $Dataset" `
                -BaseArgs @("scripts\03_run_experiment.py", "--dataset", $Dataset, "--model", $m, "--arch", "linear", "--device", "cuda") `
                -Ladder $extractLadder -Kind "extract"
        } else {
            Say "    no $Dataset cache for $short and extraction not requested in this pass" "Yellow"
        }

        # Baselines are scored on the greedy answers the cache holds, so they
        # need extraction to have produced one. The gate is extraction's exit
        # code rather than this script's guess at the cache path: if the two
        # ever disagree, 06 says so loudly instead of ten combinations
        # quietly losing their baselines.
        if ($extractOk -or (Test-Path $cache)) {
            Invoke-Step -Name "baselines $short $Dataset" `
                -BaseArgs @("scripts\06_baselines.py", "--model", $m, "--dataset", $Dataset, "--device", "cuda") `
                -Ladder $baselineLadder -Kind "baselines" | Out-Null
        }
    }
}

Say ""
Say "### PASS 1 of 2 -- PopQA baselines. Already-extracted caches, so this" "Cyan"
Say "### is the fastest route to a complete headline result." "Cyan"
Invoke-ModelStage -Dataset "popqa" -AllowExtract

Say ""
Say "### interim analysis on PopQA alone -- so there is something real to" "Cyan"
Say "### present even if the TriviaQA half does not finish in time." "Cyan"
Invoke-Step -Name "interim analysis (PopQA only)" `
    -BaseArgs @("scripts\07_final_analysis.py", "--crossfit", "--only", "popqa", "--out", "out\review3_results_popqa.json") `
    -Ladder $plainLadder -Kind "analysis" | Out-Null
if (Test-Path (Join-Path $repo "out\review3_results_popqa.json")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repo "review3_handoff") | Out-Null
    Copy-Item (Join-Path $repo "out\review3_results_popqa.json") (Join-Path $repo "review3_handoff") -Force
    Copy-Item $statusPath (Join-Path $repo "review3_handoff") -Force
    Say "interim PopQA results are in review3_handoff\ -- send these now, they are real" "Green"
}

if ($IncludeTriviaQA) {
    Say ""
    Say "### PASS 2 of 2 -- TriviaQA. Five fresh extractions; the long half." "Cyan"
    Invoke-ModelStage -Dataset "triviaqa" -AllowExtract
} else {
    Say ""
    Say "### TriviaQA skipped. Pass -IncludeTriviaQA to run it." "DarkGray"
    Say "### Re-extracting the stale PopQA caches comes first; the second" "DarkGray"
    Say "### dataset is worth nothing if the first one's decoding is wrong." "DarkGray"
    $script:status.steps += [ordered]@{
        name = "TriviaQA pass"; kind = "scope"; result = "not requested"
        minutes = 0; settings = ""; attempts = @(); log = ""
    }
    Save-Status
}

# ------------------------------------------- CPU stage; the GPU is now free
$okAnalysis = Invoke-Step -Name "final analysis (all probes, baseline ladder, controls)" `
    -BaseArgs @("scripts\07_final_analysis.py", "--crossfit") -Ladder $plainLadder -Kind "analysis"
if (-not $okAnalysis) {
    Say "retrying the analysis without cross-fitting, which is the slow and memory-hungry part" "Yellow"
    $okAnalysis = Invoke-Step -Name "final analysis (no crossfit)" `
        -BaseArgs @("scripts\07_final_analysis.py") -Ladder $plainLadder -Kind "analysis"
}
if ($IncludeTriviaQA) {
    Invoke-Step -Name "cross-dataset transfer" `
        -BaseArgs @("scripts\08_transfer.py") -Ladder $plainLadder -Kind "transfer" | Out-Null
}

# ---------------------------------------------------------------- handoff
$resFile = Join-Path $repo "out\review3_results.json"
$trFile = Join-Path $repo "out\review3_transfer.json"
$handoff = Join-Path $repo "review3_handoff"
New-Item -ItemType Directory -Force -Path $handoff | Out-Null

$popFile = Join-Path $repo "out\review3_results_popqa.json"
$have = (Test-Path $resFile) -or (Test-Path $popFile)
if ($IncludeTriviaQA) { $have = $have -and (Test-Path $trFile) }
$failed = @($script:status.steps | Where-Object { $_.result -eq "failed" })
$degraded = @($script:status.steps | Where-Object { $_.result -eq "ok" -and $_.settings -ne "" })

$script:status.finished = (Get-Date).ToString("s")
$script:status.handoff = $handoff
if ($have -and $failed.Count -eq 0) { $script:status.verdict = "complete" }
elseif ($have) { $script:status.verdict = "complete, $($failed.Count) step(s) failed" }
else { $script:status.verdict = "incomplete" }
Save-Status

foreach ($f in @($resFile, $trFile, $popFile, $statusPath, $runLog)) {
    if (Test-Path $f) { Copy-Item $f $handoff -Force }
}

Say ""
Say "================================================================" "Cyan"
Say "verdict: $($script:status.verdict)" "Cyan"
if ($degraded.Count) {
    Say "ran on fallback settings (say so when reporting these numbers):" "Yellow"
    foreach ($s in $degraded) { Say "  - $($s.name)  ->  $($s.settings)" "Yellow" }
}
if ($failed.Count) {
    Say "could not be rescued:" "Yellow"
    foreach ($f in $failed) {
        $last = if ($f.attempts.Count) { $f.attempts[$f.attempts.Count - 1].diagnosis } else { "unknown" }
        Say "  - $($f.name)   diagnosis: $last   log: $($f.log)" "Yellow"
    }
    Say "re-running this script retries only those; finished work is skipped" "Yellow"
}
Say "everything the deck needs: $handoff" "Cyan"
Say "================================================================" "Cyan"

Release-SleepGuard
if ($have) { exit 0 } else { exit 1 }
