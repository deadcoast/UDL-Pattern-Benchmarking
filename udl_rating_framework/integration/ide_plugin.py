"""
IDE plugin manager for UDL Rating Framework.

Provides integration with popular IDEs and editors for real-time UDL quality feedback.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class PluginConfig:
    """Configuration for IDE plugins."""

    enable_real_time_checking: bool = True
    quality_threshold: float = 0.7
    show_detailed_metrics: bool = True
    auto_save_reports: bool = False
    report_directory: Optional[Path] = None
    update_interval: float = 1.0  # seconds
    max_file_size_mb: int = 10


class IDEPluginManager:
    """
    Manager for IDE plugin integrations.

    Supports:
    - VS Code extension
    - IntelliJ IDEA plugin
    - Vim/Neovim plugin
    - Emacs package
    - Sublime Text package
    """

    def __init__(self, config: Optional[PluginConfig] = None):
        """Initialize IDE plugin manager."""
        self.config = config or PluginConfig()

        # Plugin templates and configurations
        self.plugin_templates = {
            "vscode": self._get_vscode_template(),
            "intellij": self._get_intellij_template(),
            "vim": self._get_vim_template(),
            "emacs": self._get_emacs_template(),
            "sublime": self._get_sublime_template(),
        }

    def generate_vscode_extension(self, output_dir: Path) -> Path:
        """
        Generate VS Code extension for UDL quality checking.

        Args:
            output_dir: Directory to create extension in

        Returns:
            Path to generated extension directory
        """
        extension_dir = output_dir / "udl-rating-vscode"
        extension_dir.mkdir(parents=True, exist_ok=True)

        # Create package.json
        package_json = {
            "name": "udl-rating",
            "displayName": "UDL Rating Framework",
            "description": "Real-time UDL quality checking and analysis",
            "version": "1.0.0",
            "engines": {"vscode": "^1.60.0"},
            "categories": ["Linters", "Other"],
            "activationEvents": [
                "onLanguage:udl",
                "onLanguage:dsl",
                "onLanguage:grammar",
            ],
            "main": "./out/extension.js",
            "contributes": {
                "languages": [
                    {
                        "id": "udl",
                        "aliases": ["UDL", "udl"],
                        "extensions": [".udl", ".dsl", ".grammar", ".ebnf"],
                        "configuration": "./language-configuration.json",
                    }
                ],
                "grammars": [
                    {
                        "language": "udl",
                        "scopeName": "source.udl",
                        "path": "./syntaxes/udl.tmGrammar.json",
                    }
                ],
                "commands": [
                    {
                        "command": "udl-rating.checkQuality",
                        "title": "Check UDL Quality",
                    },
                    {
                        "command": "udl-rating.showReport",
                        "title": "Show Quality Report",
                    },
                    {
                        "command": "udl-rating.toggleRealTime",
                        "title": "Toggle Real-time Checking",
                    },
                ],
                "configuration": {
                    "title": "UDL Rating",
                    "properties": {
                        "udl-rating.enableRealTime": {
                            "type": "boolean",
                            "default": True,
                            "description": "Enable real-time quality checking",
                        },
                        "udl-rating.qualityThreshold": {
                            "type": "number",
                            "default": 0.7,
                            "description": "Minimum quality threshold for warnings",
                        },
                        "udl-rating.showDetailedMetrics": {
                            "type": "boolean",
                            "default": True,
                            "description": "Show detailed metric information",
                        },
                    },
                },
            },
            "scripts": {
                "vscode:prepublish": "npm run compile",
                "compile": "tsc -p ./",
                "watch": "tsc -watch -p ./",
            },
            "devDependencies": {
                "@types/vscode": "^1.60.0",
                "@types/node": "14.x",
                "typescript": "^4.4.4",
            },
        }

        with open(extension_dir / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)

        # Create TypeScript extension code
        extension_ts = """
import * as vscode from 'vscode';
import { spawn } from 'child_process';
import * as path from 'path';

let diagnosticCollection: vscode.DiagnosticCollection;
let statusBarItem: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
    console.log('UDL Rating extension is now active');
    
    // Create diagnostic collection
    diagnosticCollection = vscode.languages.createDiagnosticCollection('udl-rating');
    context.subscriptions.push(diagnosticCollection);
    
    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'udl-rating.showReport';
    context.subscriptions.push(statusBarItem);
    
    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('udl-rating.checkQuality', checkQuality),
        vscode.commands.registerCommand('udl-rating.showReport', showReport),
        vscode.commands.registerCommand('udl-rating.toggleRealTime', toggleRealTime)
    );
    
    // Register document change handlers
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(onDocumentChange),
        vscode.workspace.onDidSaveTextDocument(onDocumentSave),
        vscode.window.onDidChangeActiveTextEditor(onActiveEditorChange)
    );
    
    // Check current document
    if (vscode.window.activeTextEditor) {
        checkDocumentQuality(vscode.window.activeTextEditor.document);
    }
}

async function checkQuality() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor');
        return;
    }
    
    await checkDocumentQuality(editor.document);
}

async function checkDocumentQuality(document: vscode.TextDocument) {
    if (!isUDLDocument(document)) {
        return;
    }
    
    const config = vscode.workspace.getConfiguration('udl-rating');
    const threshold = config.get<number>('qualityThreshold', 0.7);
    
    try {
        statusBarItem.text = '$(sync~spin) Checking UDL quality...';
        statusBarItem.show();
        
        const result = await runUDLRating(document);
        
        // Update diagnostics
        const diagnostics = createDiagnostics(result, threshold);
        diagnosticCollection.set(document.uri, diagnostics);
        
        // Update status bar
        const score = result.overall_score || 0;
        const emoji = score >= threshold ? '✅' : '⚠️';
        statusBarItem.text = `${emoji} UDL Quality: ${score.toFixed(3)}`;
        
    } catch (error) {
        console.error('Error checking UDL quality:', error);
        statusBarItem.text = '❌ UDL Quality: Error';
        
        const diagnostic = new vscode.Diagnostic(
            new vscode.Range(0, 0, 0, 0),
            `UDL quality check failed: ${error}`,
            vscode.DiagnosticSeverity.Error
        );
        diagnosticCollection.set(document.uri, [diagnostic]);
    }
}

function runUDLRating(document: vscode.TextDocument): Promise<any> {
    return new Promise((resolve, reject) => {
        const tempFile = path.join(__dirname, 'temp_udl.udl');
        require('fs').writeFileSync(tempFile, document.getText());
        
        const process = spawn('udl-rating', ['rate', tempFile, '--format', 'json']);
        
        let output = '';
        let error = '';
        
        process.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        process.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        process.on('close', (code) => {
            // Clean up temp file
            try {
                require('fs').unlinkSync(tempFile);
            } catch (e) {}
            
            if (code === 0) {
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (e) {
                    reject(`Failed to parse output: ${e}`);
                }
            } else {
                reject(`Process exited with code ${code}: ${error}`);
            }
        });
    });
}

function createDiagnostics(result: any, threshold: number): vscode.Diagnostic[] {
    const diagnostics: vscode.Diagnostic[] = [];
    
    // Overall quality diagnostic
    if (result.overall_score < threshold) {
        const diagnostic = new vscode.Diagnostic(
            new vscode.Range(0, 0, 0, 0),
            `UDL quality score ${result.overall_score.toFixed(3)} is below threshold ${threshold}`,
            vscode.DiagnosticSeverity.Warning
        );
        diagnostic.source = 'udl-rating';
        diagnostics.push(diagnostic);
    }
    
    // Metric-specific diagnostics
    if (result.metric_scores) {
        for (const [metric, score] of Object.entries(result.metric_scores)) {
            if (score < 0.5) {
                const diagnostic = new vscode.Diagnostic(
                    new vscode.Range(0, 0, 0, 0),
                    `Low ${metric} score: ${score.toFixed(3)}`,
                    vscode.DiagnosticSeverity.Information
                );
                diagnostic.source = 'udl-rating';
                diagnostics.push(diagnostic);
            }
        }
    }
    
    return diagnostics;
}

function isUDLDocument(document: vscode.TextDocument): boolean {
    const udlExtensions = ['.udl', '.dsl', '.grammar', '.ebnf'];
    const ext = path.extname(document.fileName).toLowerCase();
    return udlExtensions.includes(ext) || document.languageId === 'udl';
}

function onDocumentChange(event: vscode.TextDocumentChangeEvent) {
    const config = vscode.workspace.getConfiguration('udl-rating');
    if (config.get<boolean>('enableRealTime', true)) {
        // Debounce document changes
        setTimeout(() => {
            checkDocumentQuality(event.document);
        }, 1000);
    }
}

function onDocumentSave(document: vscode.TextDocument) {
    checkDocumentQuality(document);
}

function onActiveEditorChange(editor: vscode.TextEditor | undefined) {
    if (editor) {
        checkDocumentQuality(editor.document);
    } else {
        statusBarItem.hide();
    }
}

async function showReport() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || !isUDLDocument(editor.document)) {
        vscode.window.showWarningMessage('No UDL document active');
        return;
    }
    
    try {
        const result = await runUDLRating(editor.document);
        
        // Create and show report in new document
        const reportContent = createReportContent(result);
        const doc = await vscode.workspace.openTextDocument({
            content: reportContent,
            language: 'markdown'
        });
        await vscode.window.showTextDocument(doc);
        
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to generate report: ${error}`);
    }
}

function createReportContent(result: any): string {
    let content = '# UDL Quality Report\\n\\n';
    
    content += `**Overall Score**: ${result.overall_score?.toFixed(3) || 'N/A'}\\n`;
    content += `**Confidence**: ${result.confidence?.toFixed(3) || 'N/A'}\\n\\n`;
    
    if (result.metric_scores) {
        content += '## Metric Scores\\n\\n';
        for (const [metric, score] of Object.entries(result.metric_scores)) {
            const emoji = score >= 0.7 ? '✅' : score >= 0.5 ? '⚠️' : '❌';
            content += `- ${emoji} **${metric}**: ${score.toFixed(3)}\\n`;
        }
    }
    
    return content;
}

function toggleRealTime() {
    const config = vscode.workspace.getConfiguration('udl-rating');
    const current = config.get<boolean>('enableRealTime', true);
    config.update('enableRealTime', !current, vscode.ConfigurationTarget.Global);
    
    const status = !current ? 'enabled' : 'disabled';
    vscode.window.showInformationMessage(`Real-time UDL checking ${status}`);
}

export function deactivate() {
    if (diagnosticCollection) {
        diagnosticCollection.dispose();
    }
    if (statusBarItem) {
        statusBarItem.dispose();
    }
}
"""

        # Create src directory and extension.ts
        src_dir = extension_dir / "src"
        src_dir.mkdir(exist_ok=True)
        with open(src_dir / "extension.ts", "w") as f:
            f.write(extension_ts)

        # Create tsconfig.json
        tsconfig = {
            "compilerOptions": {
                "module": "commonjs",
                "target": "es6",
                "outDir": "out",
                "lib": ["es6"],
                "sourceMap": True,
                "rootDir": "src",
                "strict": True,
            },
            "exclude": ["node_modules", ".vscode-test"],
        }

        with open(extension_dir / "tsconfig.json", "w") as f:
            json.dump(tsconfig, f, indent=2)

        # Create language configuration
        lang_config = {
            "comments": {"lineComment": "#", "blockComment": ["/*", "*/"]},
            "brackets": [["{", "}"], ["[", "]"], ["(", ")"]],
            "autoClosingPairs": [
                ["{", "}"],
                ["[", "]"],
                ["(", ")"],
                ['"', '"'],
                ["'", "'"],
            ],
            "surroundingPairs": [
                ["{", "}"],
                ["[", "]"],
                ["(", ")"],
                ['"', '"'],
                ["'", "'"],
            ],
        }

        with open(extension_dir / "language-configuration.json", "w") as f:
            json.dump(lang_config, f, indent=2)

        # Create syntax highlighting
        syntaxes_dir = extension_dir / "syntaxes"
        syntaxes_dir.mkdir(exist_ok=True)

        syntax_grammar = {
            "$schema": "https://raw.githubusercontent.com/martinring/tmlanguage/master/tmlanguage.json",
            "name": "UDL",
            "patterns": [
                {"include": "#keywords"},
                {"include": "#strings"},
                {"include": "#comments"},
            ],
            "repository": {
                "keywords": {
                    "patterns": [
                        {
                            "name": "keyword.control.udl",
                            "match": "\\b(rule|token|grammar|import|export)\\b",
                        }
                    ]
                },
                "strings": {
                    "name": "string.quoted.double.udl",
                    "begin": '"',
                    "end": '"',
                    "patterns": [
                        {"name": "constant.character.escape.udl", "match": "\\\\."}
                    ],
                },
                "comments": {
                    "patterns": [
                        {"name": "comment.line.number-sign.udl", "match": "#.*$"}
                    ]
                },
            },
            "scopeName": "source.udl",
        }

        with open(syntaxes_dir / "udl.tmGrammar.json", "w") as f:
            json.dump(syntax_grammar, f, indent=2)

        logger.info(f"VS Code extension generated at {extension_dir}")
        return extension_dir

    def generate_intellij_plugin(self, output_dir: Path) -> Path:
        """Generate IntelliJ IDEA plugin for UDL quality checking."""
        plugin_dir = output_dir / "udl-rating-intellij"
        plugin_dir.mkdir(parents=True, exist_ok=True)

        # Create plugin.xml
        plugin_xml = """
<idea-plugin>
    <id>com.udlrating.intellij</id>
    <name>UDL Rating Framework</name>
    <version>1.0.0</version>
    <vendor email="support@udlrating.com" url="https://udlrating.com">UDL Rating</vendor>
    
    <description><![CDATA[
        Real-time UDL quality checking and analysis for IntelliJ IDEA.
        
        Features:
        - Real-time quality analysis
        - Detailed metric reporting
        - Code inspections and quick fixes
        - Integration with build tools
    ]]></description>
    
    <change-notes><![CDATA[
        Initial release with basic UDL quality checking functionality.
    ]]></change-notes>
    
    <idea-version since-build="203"/>
    
    <depends>com.intellij.modules.platform</depends>
    <depends>com.intellij.modules.lang</depends>
    
    <extensions defaultExtensionNs="com.intellij">
        <!-- File type -->
        <fileType name="UDL" implementationClass="com.udlrating.intellij.UDLFileType" 
                  fieldName="INSTANCE" language="UDL" extensions="udl;dsl;grammar;ebnf"/>
        
        <!-- Language -->
        <lang.parserDefinition language="UDL" 
                              implementationClass="com.udlrating.intellij.UDLParserDefinition"/>
        
        <!-- Syntax highlighter -->
        <lang.syntaxHighlighterFactory language="UDL" 
                                      implementationClass="com.udlrating.intellij.UDLSyntaxHighlighterFactory"/>
        
        <!-- Inspections -->
        <localInspection language="UDL" displayName="UDL Quality Check"
                        groupName="UDL" enabledByDefault="true"
                        implementationClass="com.udlrating.intellij.UDLQualityInspection"/>
        
        <!-- Annotator -->
        <annotator language="UDL" implementationClass="com.udlrating.intellij.UDLAnnotator"/>
        
        <!-- Tool window -->
        <toolWindow id="UDL Quality" secondary="true" anchor="bottom"
                   factoryClass="com.udlrating.intellij.UDLQualityToolWindowFactory"/>
    </extensions>
    
    <actions>
        <action id="UDL.CheckQuality" class="com.udlrating.intellij.CheckQualityAction"
                text="Check UDL Quality" description="Check quality of current UDL file">
            <add-to-group group-id="EditorPopupMenu" anchor="last"/>
            <keyboard-shortcut keymap="$default" first-keystroke="ctrl alt Q"/>
        </action>
    </actions>
</idea-plugin>
"""

        with open(plugin_dir / "plugin.xml", "w") as f:
            f.write(plugin_xml)

        # Create build.gradle
        build_gradle = '''
plugins {
    id 'java'
    id 'org.jetbrains.intellij' version '1.8.0'
}

group 'com.udlrating'
version '1.0.0'

repositories {
    mavenCentral()
}

intellij {
    version '2020.3'
    plugins = ['java']
}

patchPluginXml {
    changeNotes """
        Initial release with basic UDL quality checking functionality.
    """
}

tasks.withType(JavaCompile) {
    options.encoding = 'UTF-8'
}
'''

        with open(plugin_dir / "build.gradle", "w") as f:
            f.write(build_gradle)

        # Create Java source structure
        java_dir = (
            plugin_dir / "src" / "main" / "java" / "com" / "udlrating" / "intellij"
        )
        java_dir.mkdir(parents=True, exist_ok=True)

        # Create basic Java classes (simplified)
        file_type_java = """
package com.udlrating.intellij;

import com.intellij.openapi.fileTypes.LanguageFileType;
import com.intellij.openapi.util.IconLoader;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.swing.*;

public class UDLFileType extends LanguageFileType {
    public static final UDLFileType INSTANCE = new UDLFileType();

    private UDLFileType() {
        super(UDLLanguage.INSTANCE);
    }

    @NotNull
    @Override
    public String getName() {
        return "UDL";
    }

    @NotNull
    @Override
    public String getDescription() {
        return "User Defined Language file";
    }

    @NotNull
    @Override
    public String getDefaultExtension() {
        return "udl";
    }

    @Nullable
    @Override
    public Icon getIcon() {
        return IconLoader.getIcon("/icons/udl.png", UDLFileType.class);
    }
}
"""

        with open(java_dir / "UDLFileType.java", "w") as f:
            f.write(file_type_java)

        logger.info(f"IntelliJ plugin generated at {plugin_dir}")
        return plugin_dir

    def generate_vim_plugin(self, output_dir: Path) -> Path:
        """Generate Vim/Neovim plugin for UDL quality checking."""
        plugin_dir = output_dir / "udl-rating-vim"
        plugin_dir.mkdir(parents=True, exist_ok=True)

        # Create plugin structure
        (plugin_dir / "plugin").mkdir(exist_ok=True)
        (plugin_dir / "autoload").mkdir(exist_ok=True)
        (plugin_dir / "syntax").mkdir(exist_ok=True)
        (plugin_dir / "ftdetect").mkdir(exist_ok=True)

        # Create main plugin file
        plugin_vim = """
" UDL Rating Framework plugin for Vim/Neovim
" Maintainer: UDL Rating Team

if exists('g:loaded_udl_rating')
    finish
endif
let g:loaded_udl_rating = 1

" Configuration
if !exists('g:udl_rating_enable_realtime')
    let g:udl_rating_enable_realtime = 1
endif

if !exists('g:udl_rating_threshold')
    let g:udl_rating_threshold = 0.7
endif

if !exists('g:udl_rating_command')
    let g:udl_rating_command = 'udl-rating'
endif

" Commands
command! UDLCheckQuality call udl_rating#check_quality()
command! UDLShowReport call udl_rating#show_report()
command! UDLToggleRealtime call udl_rating#toggle_realtime()

" Auto commands
augroup UDLRating
    autocmd!
    if g:udl_rating_enable_realtime
        autocmd BufWritePost *.udl,*.dsl,*.grammar,*.ebnf call udl_rating#check_quality()
        autocmd TextChanged,TextChangedI *.udl,*.dsl,*.grammar,*.ebnf call udl_rating#check_quality_delayed()
    endif
augroup END

" Key mappings
nnoremap <leader>uq :UDLCheckQuality<CR>
nnoremap <leader>ur :UDLShowReport<CR>
nnoremap <leader>ut :UDLToggleRealtime<CR>
"""

        with open(plugin_dir / "plugin" / "udl_rating.vim", "w") as f:
            f.write(plugin_vim)

        # Create autoload functions
        autoload_vim = """
" UDL Rating autoload functions

let s:timer_id = -1
let s:last_result = {}

function! udl_rating#check_quality()
    if !executable(g:udl_rating_command)
        echohl ErrorMsg
        echo "UDL Rating command not found: " . g:udl_rating_command
        echohl None
        return
    endif
    
    let l:filename = expand('%:p')
    if empty(l:filename) || !filereadable(l:filename)
        return
    endif
    
    " Check if file is UDL
    if !s:is_udl_file(l:filename)
        return
    endif
    
    " Run quality check
    let l:cmd = g:udl_rating_command . ' rate "' . l:filename . '" --format json'
    let l:output = system(l:cmd)
    
    if v:shell_error == 0
        try
            let l:result = json_decode(l:output)
            let s:last_result = l:result
            call s:update_signs(l:result)
            call s:update_statusline(l:result)
        catch
            echohl ErrorMsg
            echo "Failed to parse UDL rating output"
            echohl None
        endtry
    else
        echohl ErrorMsg
        echo "UDL rating failed: " . l:output
        echohl None
    endif
endfunction

function! udl_rating#check_quality_delayed()
    " Cancel previous timer
    if s:timer_id != -1
        call timer_stop(s:timer_id)
    endif
    
    " Start new timer
    let s:timer_id = timer_start(1000, function('s:delayed_check'))
endfunction

function! s:delayed_check(timer)
    let s:timer_id = -1
    call udl_rating#check_quality()
endfunction

function! udl_rating#show_report()
    if empty(s:last_result)
        echo "No quality data available. Run :UDLCheckQuality first."
        return
    endif
    
    " Create report buffer
    let l:bufname = 'UDL Quality Report'
    let l:bufnr = bufnr(l:bufname)
    
    if l:bufnr == -1
        execute 'new ' . l:bufname
        setlocal buftype=nofile
        setlocal bufhidden=wipe
        setlocal noswapfile
        setlocal filetype=markdown
    else
        execute 'buffer ' . l:bufnr
        %delete _
    endif
    
    " Generate report content
    call append(0, '# UDL Quality Report')
    call append(1, '')
    call append(2, '**File**: ' . expand('#:p'))
    call append(3, '**Overall Score**: ' . printf('%.3f', get(s:last_result, 'overall_score', 0)))
    call append(4, '**Confidence**: ' . printf('%.3f', get(s:last_result, 'confidence', 0)))
    call append(5, '')
    call append(6, '## Metric Scores')
    call append(7, '')
    
    let l:line = 8
    if has_key(s:last_result, 'metric_scores')
        for [l:metric, l:score] in items(s:last_result.metric_scores)
            let l:emoji = l:score >= 0.7 ? '✅' : l:score >= 0.5 ? '⚠️' : '❌'
            call append(l:line, '- ' . l:emoji . ' **' . l:metric . '**: ' . printf('%.3f', l:score))
            let l:line += 1
        endfor
    endif
    
    " Go to top
    normal! gg
endfunction

function! udl_rating#toggle_realtime()
    let g:udl_rating_enable_realtime = !g:udl_rating_enable_realtime
    
    if g:udl_rating_enable_realtime
        echo "UDL real-time checking enabled"
        augroup UDLRating
            autocmd BufWritePost *.udl,*.dsl,*.grammar,*.ebnf call udl_rating#check_quality()
            autocmd TextChanged,TextChangedI *.udl,*.dsl,*.grammar,*.ebnf call udl_rating#check_quality_delayed()
        augroup END
    else
        echo "UDL real-time checking disabled"
        augroup UDLRating
            autocmd!
        augroup END
    endif
endfunction

function! s:is_udl_file(filename)
    let l:ext = tolower(fnamemodify(a:filename, ':e'))
    return index(['udl', 'dsl', 'grammar', 'ebnf'], l:ext) >= 0
endfunction

function! s:update_signs(result)
    " Clear existing signs
    sign unplace *
    
    let l:score = get(a:result, 'overall_score', 0)
    let l:sign_name = l:score >= g:udl_rating_threshold ? 'UDLGood' : 'UDLWarning'
    
    " Define signs if not already defined
    if !exists('s:signs_defined')
        sign define UDLGood text=✅ texthl=DiffAdd
        sign define UDLWarning text=⚠️ texthl=DiffChange
        sign define UDLError text=❌ texthl=DiffDelete
        let s:signs_defined = 1
    endif
    
    " Place sign on first line
    execute 'sign place 1 line=1 name=' . l:sign_name . ' buffer=' . bufnr('%')
endfunction

function! s:update_statusline(result)
    let l:score = get(a:result, 'overall_score', 0)
    let l:emoji = l:score >= g:udl_rating_threshold ? '✅' : '⚠️'
    let b:udl_quality_status = l:emoji . ' UDL: ' . printf('%.3f', l:score)
endfunction
"""

        with open(plugin_dir / "autoload" / "udl_rating.vim", "w") as f:
            f.write(autoload_vim)

        # Create syntax file
        syntax_vim = """
" Vim syntax file for UDL
" Language: User Defined Language
" Maintainer: UDL Rating Team

if exists("b:current_syntax")
    finish
endif

" Keywords
syn keyword udlKeyword rule token grammar import export
syn keyword udlType string number boolean
syn keyword udlOperator -> | + * ? 

" Strings
syn region udlString start='"' end='"' contains=udlEscape
syn match udlEscape '\\.' contained

" Comments
syn match udlComment '#.*$'
syn region udlBlockComment start='/\\*' end='\\*/'

" Identifiers
syn match udlIdentifier '[a-zA-Z_][a-zA-Z0-9_]*'

" Highlighting
hi def link udlKeyword Keyword
hi def link udlType Type
hi def link udlOperator Operator
hi def link udlString String
hi def link udlEscape Special
hi def link udlComment Comment
hi def link udlBlockComment Comment
hi def link udlIdentifier Identifier

let b:current_syntax = "udl"
"""

        with open(plugin_dir / "syntax" / "udl.vim", "w") as f:
            f.write(syntax_vim)

        # Create file type detection
        ftdetect_vim = """
" UDL file type detection
au BufRead,BufNewFile *.udl,*.dsl,*.grammar,*.ebnf set filetype=udl
"""

        with open(plugin_dir / "ftdetect" / "udl.vim", "w") as f:
            f.write(ftdetect_vim)

        logger.info(f"Vim plugin generated at {plugin_dir}")
        return plugin_dir

    def _get_vscode_template(self) -> Dict[str, Any]:
        """Get VS Code extension template."""
        return {
            "type": "vscode",
            "files": ["package.json", "src/extension.ts", "tsconfig.json"],
        }

    def _get_intellij_template(self) -> Dict[str, Any]:
        """Get IntelliJ plugin template."""
        return {
            "type": "intellij",
            "files": ["plugin.xml", "build.gradle", "src/main/java/**/*.java"],
        }

    def _get_vim_template(self) -> Dict[str, Any]:
        """Get Vim plugin template."""
        return {
            "type": "vim",
            "files": [
                "plugin/*.vim",
                "autoload/*.vim",
                "syntax/*.vim",
                "ftdetect/*.vim",
            ],
        }

    def _get_emacs_template(self) -> Dict[str, Any]:
        """Get Emacs package template."""
        return {"type": "emacs", "files": ["udl-rating.el"]}

    def _get_sublime_template(self) -> Dict[str, Any]:
        """Get Sublime Text package template."""
        return {
            "type": "sublime",
            "files": ["UDL.sublime-syntax", "UDL.sublime-settings"],
        }

    def install_plugin(self, plugin_type: str, plugin_dir: Path) -> bool:
        """
        Install plugin for specified IDE.

        Args:
            plugin_type: Type of plugin ('vscode', 'intellij', 'vim', etc.)
            plugin_dir: Directory containing plugin files

        Returns:
            True if installation successful
        """
        try:
            if plugin_type == "vscode":
                return self._install_vscode_extension(plugin_dir)
            elif plugin_type == "intellij":
                return self._install_intellij_plugin(plugin_dir)
            elif plugin_type == "vim":
                return self._install_vim_plugin(plugin_dir)
            else:
                logger.error(f"Unsupported plugin type: {plugin_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to install {plugin_type} plugin: {e}")
            return False

    def _install_vscode_extension(self, plugin_dir: Path) -> bool:
        """Install VS Code extension."""
        try:
            # Build extension
            subprocess.run(["npm", "install"], cwd=plugin_dir, check=True)
            subprocess.run(["npm", "run", "compile"],
                           cwd=plugin_dir, check=True)

            # Package extension
            subprocess.run(["vsce", "package"], cwd=plugin_dir, check=True)

            # Install extension
            vsix_files = list(plugin_dir.glob("*.vsix"))
            if vsix_files:
                subprocess.run(
                    ["code", "--install-extension", str(vsix_files[0])], check=True
                )
                logger.info("VS Code extension installed successfully")
                return True
            else:
                logger.error("No VSIX file found")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install VS Code extension: {e}")
            return False

    def _install_intellij_plugin(self, plugin_dir: Path) -> bool:
        """Install IntelliJ plugin."""
        try:
            # Build plugin
            subprocess.run(["./gradlew", "buildPlugin"],
                           cwd=plugin_dir, check=True)

            logger.info("IntelliJ plugin built successfully")
            logger.info(
                "Install manually through IntelliJ IDEA plugin manager")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build IntelliJ plugin: {e}")
            return False

    def _install_vim_plugin(self, plugin_dir: Path) -> bool:
        """Install Vim plugin."""
        try:
            # Determine Vim config directory
            vim_config_dirs = [
                Path.home() / ".vim", Path.home() / ".config" / "nvim"]

            vim_config_dir = None
            for config_dir in vim_config_dirs:
                if config_dir.exists():
                    vim_config_dir = config_dir
                    break

            if not vim_config_dir:
                # Create .vim directory
                vim_config_dir = Path.home() / ".vim"
                vim_config_dir.mkdir(exist_ok=True)

            # Copy plugin files
            for subdir in ["plugin", "autoload", "syntax", "ftdetect"]:
                src_dir = plugin_dir / subdir
                dst_dir = vim_config_dir / subdir

                if src_dir.exists():
                    dst_dir.mkdir(exist_ok=True)
                    for file_path in src_dir.glob("*"):
                        shutil.copy2(file_path, dst_dir)

            logger.info(f"Vim plugin installed to {vim_config_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to install Vim plugin: {e}")
            return False


def main():
    """CLI entry point for IDE plugin management."""
    import argparse

    parser = argparse.ArgumentParser(description="UDL IDE Plugin Manager")
    parser.add_argument("action", choices=["generate", "install"])
    parser.add_argument(
        "--plugin-type",
        choices=["vscode", "intellij", "vim", "emacs", "sublime"],
        required=True,
        help="Type of plugin to generate/install",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for generated plugins",
    )
    parser.add_argument(
        "--plugin-dir",
        type=Path,
        help="Directory containing plugin files (for install action)",
    )

    args = parser.parse_args()

    manager = IDEPluginManager()

    if args.action == "generate":
        if args.plugin_type == "vscode":
            plugin_dir = manager.generate_vscode_extension(args.output_dir)
        elif args.plugin_type == "intellij":
            plugin_dir = manager.generate_intellij_plugin(args.output_dir)
        elif args.plugin_type == "vim":
            plugin_dir = manager.generate_vim_plugin(args.output_dir)
        else:
            print(
                f"Plugin generation for {args.plugin_type} not yet implemented")
            return

        print(f"Plugin generated at: {plugin_dir}")

    elif args.action == "install":
        if not args.plugin_dir:
            print("--plugin-dir is required for install action")
            return

        success = manager.install_plugin(args.plugin_type, args.plugin_dir)
        if success:
            print(f"{args.plugin_type} plugin installed successfully")
        else:
            print(f"Failed to install {args.plugin_type} plugin")
            exit(1)


if __name__ == "__main__":
    main()
