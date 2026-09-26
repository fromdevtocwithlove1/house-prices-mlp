$ErrorActionPreference = 'Stop'
$reports = Split-Path -Parent $MyInvocation.MyCommand.Path
$docx = Join-Path $reports 'bao_cao_crisp_dm.docx'
$pdf = Join-Path $reports 'bao_cao_crisp_dm.pdf'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($docx)
    $document.Repaginate()
    if ($document.TablesOfContents.Count -ne 1) {
        throw "Expected one Word table of contents; found $($document.TablesOfContents.Count)"
    }
    $toc = $document.TablesOfContents.Item(1)
    $toc.Update()
    $document.Fields.Update() | Out-Null
    $document.Repaginate()
    $toc.UpdatePageNumbers()
    $document.Save()
    $document.ExportAsFixedFormat($pdf, 17)
    "DOCX=$docx"
    "PDF=$pdf"
    "PAGES=$($document.ComputeStatistics(2))"
    "TOC_BEGIN"
    $toc.Range.Text
    "TOC_END"
} finally {
    if ($document -ne $null) {
        $document.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($word -ne $null) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
}
