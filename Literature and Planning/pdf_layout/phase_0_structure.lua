-- Prepare the Markdown heading structure for the standalone PDF.
local report_title = "UAV Remaining Useful Life Estimation"
local removed_document_heading = false

function Header(header)
  local text = pandoc.utils.stringify(header.content)

  -- The title is supplied as document metadata for a proper title page.
  if not removed_document_heading and header.level == 1 and text == report_title then
    removed_document_heading = true
    return {}
  end

  -- Promote the remaining headings after removing the Markdown document title.
  if header.level > 1 then
    header.level = header.level - 1
  end

  -- Major report chapters start on a fresh page.
  if header.level == 1 then
    return {
      pandoc.RawBlock("latex", "\\clearpage"),
      header,
    }
  end

  return header
end

