" function! s:GPTTryUpdateDiff(target_bufnr)
"     if exists('*GPTUpdateDiff')
"         let savedview = winsaveview()
"         let ei_bak = &eventignore
"         set eventignore=all
"        
"         let new_content = getbufline(a:target_bufnr,'^','$')
"         silent undo
"         let old_content = getbufline(a:target_bufnr,'^','$')
"         silent redo
"         if new_content != old_content
"             call GPTUpdateDiff(new_content, old_content)
"         endif
"
"         call winrestview(savedview)
"         let &eventignore = ei_bak
"     endif
" endfunction

augroup gpt4o
    autocmd!
    autocmd! InsertLeave * GPTHold!
    autocmd! BufEnter,CursorHold,BufLeave,BufWritePost * GPTHold
augroup end
